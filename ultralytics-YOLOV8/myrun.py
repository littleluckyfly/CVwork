# from ultralytics import YOLO
# import warnings
#
# warnings.filterwarnings('ignore')
# if __name__ == '__main__':
#     model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
#
#     # Train the model
#     results = model.train(data='wutian.yaml', epochs=100, imgsz=640)
#
#     # # 模型验证
#     # model = YOLO('runs/detect/train2/weights/best.pt')
#     # model.val(**{'data': 'ultralytics/cfg/datasets/wutian.yaml'})
#
#     # # 模型推理
#     # model = YOLO('runs/detect/train3/weights/best.pt')
#     # model.predict(source='datasets/UG_yolov8/images/test', **{'save': True})
#
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.yaml')
    # model.load('yolov8n.pt')  # loading pretrain weights
    model.train(data=r'Road.yaml',
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                cache=False,
                imgsz=640,
                epochs=100,
                # batch=8,
                # workers=4,
                single_cls=False,  # 是否是单类别检测
                close_mosaic=10,
                optimizer='SGD',  # using SGD
                # resume='runs/detect/train5/weights/last.pt',  # 如过想续训就设置last.pt的地址
                amp=False,  # 如果出现训练损失为Nan可以关闭amp
                )
