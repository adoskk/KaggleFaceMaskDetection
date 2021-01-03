import os
from utilities.data_utils.Dataset import FacialDataset, get_transform
from utilities.utils import collate_fn
from utilities.train_eval.engine import train_one_epoch, evaluate, get_model_result
import glob

import nvidia_smi # for python 3, you need nvidia-ml-py3 library

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead

save_model_folder = 'model'
output_image_folder = 'output'
num_classes = 3  # 2 class (person) + background
batch_size = 5
num_epochs = 20


if __name__ == "__main__":
    
    torch.cuda.empty_cache()
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print("Total memory:", info.total)
    print("Free memory:", info.free)
    print("Used memory:", info.used)

    dataset_train = FacialDataset('data/train', get_transform(horizontal_flip=True))
    dataset_test = FacialDataset('data/test', get_transform(horizontal_flip=False))

    data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,
            collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn)
        
        
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    anchor_generator = AnchorGenerator(sizes=((32,), (24, ), (24, ), (16,), (8, )),
                                        aspect_ratios=([1.0, 1.0, 1.0, 1.0], 
                                                     [0.8, 1.0, 1.0, 1.0], 
                                                     [1.0, 0.8, 1.0, 1.0],
                                                     [1.0, 1.0, 1.0, 1.0],
                                                     [1.0, 1.0, 1.0, 1.0]))
    model.rpn.anchor_generator = anchor_generator
    model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
        
    for epoch in range(num_epochs):
        res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
        
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)

    print("Training complete!")
    # create output directory
    # if output directory exists, delete existing files
    if not os.path.exists(output_image_folder):
        os.mkdir(output_image_folder)
    else:
        files = glob.glob(output_image_folder + '/*')
        for f in files:
            os.remove(f)
    # write testing result to output folder
    for img_idx, batch_sampler in enumerate(data_loader_test):
        img_test = batch_sampler[0][0]
        target_test = batch_sampler[1][0]
        i = target_test["image_id"].item()
        get_model_result(img_test, model, target_test, i, device, location=output_image_folder, threshold=0.15)

    print("Testing complete!")
    
    torch.cuda.synchronize()
    # create directory for saving the model
    if not os.path.exists(save_model_folder):
        os.mkdir(save_model_folder)
    print("Saving model...")
    torch.save(model.state_dict(), os.path.join(save_model_folder, 'FaceMaskDetection_TrainEpoch_' + str(num_epochs) + '.pth'))
    print("Model saving complete!")
    nvidia_smi.nvmlShutdown()

