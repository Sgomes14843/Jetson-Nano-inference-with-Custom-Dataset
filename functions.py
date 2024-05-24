from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import numpy as np

def predict_image(net_type, model_path, label_path, image_path, output_path, size_candidate=200):

  class_names = [name.strip() for name in open(label_path).readlines()]

  if net_type == 'mb1-ssd':
      net = create_mobilenetv1_ssd(len(class_names), is_test=True)
  else:
      print("The net type is wrong")
      sys.exit(1)
  net.load(model_path)

  if net_type == 'mb1-ssd':
      predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=size_candidate)
  else:
      predictor = create_vgg_ssd_predictor(net, candidate_size=size_candidate)
  
  orig_image = cv2.imread(image_path)
  image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
  boxes, labels, probs = predictor.predict(image, 10, 0.4)

  for i in range(boxes.size(0)):
    box = boxes[i, :]
    cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
    cv2.putText(orig_image, label,
                (int(box[0]) + 20, int(box[1]) + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
  cv2.imwrite(output_path, orig_image)
  return f"Found {len(probs)} objects. The output image is {output_path}"

def predict_video(net_type, model_path, label_path, video_file, output_path, size_candidate=200):
    cap = cv2.VideoCapture(video_file)  # capture from file
    class_names = [name.strip() for name in open(label_path).readlines()]
    num_classes = len(class_names)
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    if net_type == 'mb1-ssd':
        net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    else:
        print("The net type is wrong")
        sys.exit(1)
    
    net.load(model_path)

    if net_type == 'mb1-ssd':
        predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
    else:
        print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        sys.exit(1)

    timer = Timer()
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps2 = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4',fourcc, fps2, (width,height))

    while True:
        ret, orig_image = cap.read()
        if orig_image is None:
            continue
        while(ret):
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            timer.start()
            boxes, labels, probs = predictor.predict(image, 10, 0.7)
            interval = timer.end()
            print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
            for i in range(boxes.size(0)):
                box = boxes[i, :]
                label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
                cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
                
                cv2.putText(orig_image, label,
                            (int(box[0])+20, int(box[1])+40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,  # font scale
                            (255, 0, 255),
                            2)  # line type
            # cv2.imshow('video', orig_image)
            out.write(orig_image)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print ('FPS : {0}'.format(fps))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret, orig_image = cap.read()
        break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return f"The output video is {output_path}"
