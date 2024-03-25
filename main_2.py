from ultralytics import YOLO, settings

model = YOLO('yolov8n.yaml')
# results = model.train(data='C:\\tmp\\dataset\\config.yaml', epochs=1)
# root_config = 'C:\\Users\\domin\\OneDrive\\Pulpit\\Python\\DetectionShapes\\dataset\\config.yaml'
root_config = 'E:\\USERS\\dominik.roczan\\PycharmProjects\\DetectionShapes\\dataset\\config.yaml'


if __name__ == '__main__':
    results = model.train(data=root_config, epochs=100)

#tensorboard --logdir E:\USERS\dominik.roczan\PycharmProjects\DetectionShapes\runs
