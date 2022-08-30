import torch
from torchsummary import summary
from nets.yolo import YoloBody
if __name__ == "__main__":
    
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    m       = YoloBody([[6, 7, 8], [3, 4, 5], [0, 1, 2]], 3, 'l').to(device)
    
    
    
    
    summary(net, input_size=(4, 320, 320))
