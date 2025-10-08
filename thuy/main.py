from ultralytics import YOLO
import yaml

data = {
    'path': 'D:/yolo/datasets',   
    'train': 'images',
    'val': 'images',       
    'names': {
        0: 'standing_person',
        1: 'dangerous_behavior',
        2: 'weapon',
        3: 'high_place'
    }
}
with open(r'D:\yolo\configs\data.yaml', 'w') as f:
    yaml.dump(data, f, sort_keys=False)

# 2. Train model YOLO
model = YOLO('yolo11n.pt')  

results = model.train(
    data=r'D:\yolo\configs\data.yaml',
    epochs=100,       
    imgsz=640,        
    batch=16,          
    patience=10,      
    save=True,       
    device='cpu',    
    val=False         
)
