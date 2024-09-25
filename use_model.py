import torch
import torchvision.io.image
from torchvision.transforms import ToTensor
from torchvision.transforms.v2 import Transform

#github下载，并且指定版本，如果到其他项目中自动下载
# model = torch.hub.load('ultralytics/yolov5:v5.0', 'yolov5s')

#本地加载 hubconfig.py 模型-如果当前 没有yolov5s.pt 权重也是从远程下载，本地还是需要模型的定义
model = torch.hub.load('.', 'yolov5s',  source='local')

# 手动定义-- 需要对权重进行手动修改- 参考 hubconf.py
# model = Model("models/yolov5s.yaml", 3, 80)
# state_dict= torch.load("yolov5s.pt") # 加载的dict 需要一些处理才能使用，直接加载会错误
# model.load_state_dict(state_dict)

# 设置模型为评估模式
model.eval()
img = 'data/images/bus.jpg'
results = model(img)
# 导出 onnx
input=torch.randn(1,3,640,640)
torch.onnx.export(model,args=input, f='yolov5s.onnx')

# 打印结果
results.print()
# 以列表形式获取所有检测框
detections = results.xyxy[0].numpy()  # x1, y1, x2, y2, confidence, class
for det in detections:
    x1, y1, x2, y2, conf, cls = det  # 坐标和置信度,类别
    print(f"Detected object with confidence {conf} 类别 {cls}  at position [{x1}, {y1}, {x2}, {y2}]")

