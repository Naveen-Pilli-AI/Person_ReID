
import os
import cv2
import yaml
import argparse
from datetime import datetime
import numpy as np
from scipy.spatial import distance
from tqdm.autonotebook import tqdm
from detector import object_detector
from itertools import count as while_true
# from object_detection import ObjectDetection
from feature_extraction import FeatureExtraction
from helpers import stack_images, new_coordinates_resize, setup_resolution

vid_path_1='/home/user/CV_Projects/Projects/Multipul_cameras/4_people_data/4p-c0.avi'
# cap=cv2.VideoCapture(vid_path_1)

def main(cfg):
    # Variable for save detected person
    detected_persons = {}

 
    # print(object_detection)

    # Init feature extraction
    feature_extraction = FeatureExtraction(
        onnx_path=cfg["feature_extraction_model_path"],
        device=cfg["inference_model_device"],
    )

    # Setup camera
    cam = {}
 
    videos = np.array(os.listdir(cfg["video_path"]))
    total_cam = len(videos)

    for i in range(total_cam):
        cam[f"cam_{i}"] = cv2.VideoCapture(os.path.join(cfg["video_path"], videos[i]))
        cam[f"cam_{i}"].set(3, cfg["size_each_camera_image"][0])
        cam[f"cam_{i}"].set(4, cfg["size_each_camera_image"][1])
       

   
    if cfg["save_video_camera_tracking"]:
        out = cv2.VideoWriter(
            os.path.join(
                cfg["output_path_name_save_video_camera_tracking"],
                f'{cfg["output_name_save_video_camera_tracking"]}.avi',
            ),
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),
            cfg["fps_save_video_camera_tracking"],
            setup_resolution(
                cfg["size_each_camera_image"], cfg["resize_all_camera_image"], total_cam
            ),
        )

    id = 0
    prev= datetime.today() 
    seconds = prev.timestamp() 
    # for _ in tqdm(while_true(), desc="Tracking person in progress..."):
    for _ in while_true():
        # Set up variable
        images = {}
        predicts = {}
        # curr_time=datetime.datetime.now()
        # Get camera image
        for i in range(total_cam):
            _, images[f"image_{i}"] = cam[f"cam_{i}"].read()

        # Predict person with object detection
        for i in range(total_cam):
            # pred=object_detector(images[f"image_{i}"])
            # print(pred)
            predicts[f"image_{i}"] = object_detector(images[f"image_{i}"])

            # print(predicts)

        # Resize image for display in screen
        for i in range(total_cam):
            images[f"image_{i}"] = cv2.resize(
                images[f"image_{i}"],
                cfg["size_each_camera_image"],
                interpolation=cv2.INTER_CUBIC,
            )

        for i in range(total_cam):
            # print (i)
            for predict in predicts[f"image_{i}"]:
                cls_name = tuple(predict.keys())[0]
                x1, y1, x2, y2 = predict[cls_name]["bounding_box"]

                # Person identification
                cropped_image = images[f"image_{i}"][y1:y2, x1:x2]
                extracted_features = feature_extraction.predict_img(cropped_image)[0].reshape(-1)
                # print(extracted_features)

                # Add new person if data is empty
                if not detected_persons:
                    detected_persons[f"id_{id}"] = {
                        "extracted_features": extracted_features,
                        "id": id,
                        "cls_name": cls_name,
                        "bbox": (x1, y1, x2, y2),
                        "confidence": predict[cls_name]["confidence"],
                        "color": np.random.randint(0, 255, size=3),
                    }
                    id += 1

                # print(detected_persons)    
                   
                else:
                    top1_person = np.array(
                        [
                            {
                                "id": value["id"],
                                "cls_name": value["cls_name"],
                                "color": value["color"],
                                "score": distance.cosine(
                                    np.expand_dims(
                                        np.mean(value["extracted_features"], axis=0),
                                        axis=0,
                                    )
                                    if len(value["extracted_features"]) > 1
                                    else value["extracted_features"],
                                    extracted_features,
                                ),
                            }
                            for value in detected_persons.values()
                        ]
                    )
                    top1_person = sorted(
                        top1_person, key=lambda d: d["score"], reverse=False
                    )[0]
                    # print(top1_person)
                    print(extracted_features.shape)
                    #Add data for new person or replace new bbox, confidence object detection, feature extraction embedding, and camera id for existing person
                    if top1_person["score"] < cfg["feature_extraction_threshold"]:
                        detected_persons[f"id_{top1_person['id']}"] = {
                            "extracted_features": np.vstack(
                                (
                                    detected_persons[f"id_{top1_person['id']}"][
                                        "extracted_features"
                                    ],
                                    extracted_features,
                                )
                            )
                            if detected_persons[f"id_{top1_person['id']}"][
                                "extracted_features"
                            ].shape[0]
                            < cfg["max_gallery_set_each_person"]
                            else np.vstack(
                                (
                                    extracted_features,
                                   detected_persons[f"id_{top1_person['id']}"][
                                        "extracted_features"
                                    ][1:],
                                )
                            ),
                            "id": top1_person["id"],
                            "camera_id": i,
                            "cls_name": top1_person["cls_name"],
                            "bbox": (x1, y1, x2, y2),
                            "confidence": predict[cls_name]["confidence"],
                            "color": top1_person["color"],
                        }
                    else:
                        detected_persons[f"id_{id}"] = {
                            "extracted_features": extracted_features,
                            "id": id,
                            "camera_id": i,
                            "cls_name": cls_name,
                            "bbox": (x1, y1, x2, y2),
                            "confidence": predict[cls_name]["confidence"],
                            "color": np.random.randint(0, 255, size=3),
                        }
                        id += 1


            # print(detected_persons)
            count=0
            # Draw all bbox
            try:
                for value in detected_persons.values():
                    print(value)
                    print(i)
                    print(value["camera_id"])
                    if value["camera_id"] == i:
                        cv2.rectangle(
                            images[f"image_{value['camera_id']}"],
                            value["bbox"][:2],
                            value["bbox"][2:],
                            value["color"].tolist(),
                            2,
                        )
                        cv2.putText(
                            images[f"image_{value['camera_id']}"],
                            f"{value['cls_name']} {value['id']}: {value['confidence']}",
                            (value["bbox"][0], value["bbox"][1] - 10),
                            cv2.FONT_HERSHEY_PLAIN,
                            1,
                            value["color"].tolist(),
                            2,
                        )
                        # print(curr_time)
                        # cv2.putText(images[f"image_{value['camera_id']}"],str(curr_time),)

                        cur=datetime.today() 
                        cur_seconds = cur.timestamp()
                        total_seconds=cur_seconds-seconds
                        total_seconds=round(total_seconds)
                        # print(total_seconds)
                        cv2.putText(images[f"image_{value['camera_id']}"],f"{str(total_seconds)} Sec", (500, 400), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,0,200), 2)
                        # if total_seconds==51 or total_seconds==205.09:
                        #     cv2.imwrite(f'results/image_{count}.jpg',images[f"image_{value['camera_id']}"])
                        #     print('first image')

                        # count=count+1
                        # print(total_seconds)
            except:
                pass

        # Display all cam
        if total_cam % 2 == 0:
            display_image = stack_images(
                cfg["resize_all_camera_image"],
                (
                    [images[f"image_{i}"] for i in range(0, total_cam // 2)],
                    [images[f"image_{i}"] for i in range(total_cam // 2, total_cam)],
                ),
            )
        else:
            display_image = stack_images(
                cfg["resize_all_camera_image"],
                ([images[f"image_{i}"] for i in range(total_cam)],),
            )

        if cfg["save_video_camera_tracking"]:
            out.write(display_image)
        if cfg["display_video_camera_tracking"]:
            cv2.imshow("CCTV Misale", display_image)
            if cv2.waitKey(1) == ord("q"):
                break
        # print(curr_time)

    # Release all cam
    for i in range(total_cam):
        cam[f"cam_{i}"].release()
    if cfg["save_video_camera_tracking"]:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--source-config-file",
        default="./config.yaml",
        help="Input your config.yaml file",
    )
    value_parser = parser.parse_args()

    with open(value_parser.source_config_file, "r") as f:
        file_config = yaml.safe_load(f)
    main(file_config)

