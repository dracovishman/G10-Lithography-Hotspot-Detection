import os, glob, random, argparse, json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

class ImgDataset(Dataset):
    def __init__(self, paths, labels, size=64):
        self.paths=paths; self.labels=np.asarray(labels,dtype=np.float32); self.size=size
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        with Image.open(self.paths[i]) as im:
            im=im.convert("L").resize((self.size,self.size),Image.Resampling.BILINEAR)
        x=np.asarray(im,dtype=np.float32)/255.0
        return torch.from_numpy(x[None]), torch.tensor(self.labels[i])

class BasicBlock(nn.Module):
    def __init__(self,in_ch,ch=12):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_ch,ch,3), nn.ELU(),
            nn.Conv2d(ch,ch,3), nn.ELU(),
            nn.Conv2d(ch,ch,3),
            nn.BatchNorm2d(ch), nn.ELU(),
            nn.MaxPool2d(2))
    def forward(self,x): return self.net(x)

class LightweightCNN(nn.Module):
    def __init__(self,size=64):
        super().__init__()
        self.b1=BasicBlock(1,12); self.b2=BasicBlock(12,12)
        with torch.no_grad():
            n=self.b2(self.b1(torch.zeros(1,1,size,size))).flatten(1).shape[1]
        self.fc=nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(n,10), nn.ELU(), nn.Linear(10,1))
    def forward(self,x):
        return self.fc(self.b2(self.b1(x))).squeeze(1)

def focal_loss(logits,y,alpha=0.75,gamma=2.0):
    bce=nn.functional.binary_cross_entropy_with_logits(logits,y,reduction="none")
    p=torch.sigmoid(logits)
    pt=torch.where(y>0.5,p,1-p)
    at=torch.where(y>0.5,torch.full_like(y,alpha),
                   torch.full_like(y,1-alpha))
    return at*(1-pt).pow(gamma)*bce

def get_train_files(root,b):
    q=os.path.join(root,f"iccad{b}")
    hs=glob.glob(q+"/train/train_hs/*.png")
    nhs=glob.glob(q+"/train/train_nhs/*.png")
    return hs+nhs, [1]*len(hs)+[0]*len(nhs)

def ba_score(y,p):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return 0.5*(tp/(tp+fn) + tn/(tn+fp))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--benchmark",type=int,required=True)
    ap.add_argument("--mode",choices=["bce","focal","ohem","focal_ohem"],required=True)
    ap.add_argument("--epochs",type=int,default=5)
    ap.add_argument("--batch",type=int,default=128)
    ap.add_argument("--size",type=int,default=64)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--gamma",type=float,default=2.0)
    ap.add_argument("--alpha",type=float,default=0.75)
    ap.add_argument("--hard_fraction",type=float,default=0.5)
    ap.add_argument("--outdir",default="checkpoints")
    args=ap.parse_args()

    paths,labels=get_train_files(args.root,args.benchmark)
    idx=np.arange(len(paths))
    ti,vi=train_test_split(idx,test_size=0.2,stratify=labels,random_state=SEED)

    tr=ImgDataset([paths[i] for i in ti],[labels[i] for i in ti],args.size)
    va=ImgDataset([paths[i] for i in vi],[labels[i] for i in vi],args.size)
    tl=DataLoader(tr,batch_size=args.batch,shuffle=True,num_workers=2,pin_memory=True)
    vl=DataLoader(va,batch_size=args.batch,shuffle=False,num_workers=2,pin_memory=True)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=LightweightCNN(args.size).to(device)
    opt=torch.optim.NAdam(model.parameters(),lr=args.lr)

    best_ba=-1
    best_state=None
    best_epoch=0

    for ep in range(1,args.epochs+1):
        model.train()
        for x,y in tl:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            z=model(x)

            if args.mode=="bce":
                losses=nn.functional.binary_cross_entropy_with_logits(z,y,reduction="none")
            elif args.mode=="focal":
                losses=focal_loss(z,y,args.alpha,args.gamma)
            elif args.mode=="ohem":
                losses=nn.functional.binary_cross_entropy_with_logits(z,y,reduction="none")
            else:
                losses=focal_loss(z,y,args.alpha,args.gamma)

            if args.mode in ("ohem","focal_ohem"):
                k=max(1,int(len(losses)*args.hard_fraction))
                losses=torch.topk(losses,k).values

            losses.mean().backward()
            opt.step()

        model.eval(); ys=[]; ps=[]
        with torch.inference_mode():
            for x,y in vl:
                p=(torch.sigmoid(model(x.to(device)))>=0.5).int().cpu().tolist()
                ys.extend(y.int().tolist()); ps.extend(p)

        score=ba_score(ys,ps)
        print(f"benchmark={args.benchmark} mode={args.mode} epoch={ep} val_BA={score:.6f}")

        if score>best_ba:
            best_ba=score; best_epoch=ep
            best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}

    os.makedirs(args.outdir,exist_ok=True)
    out=os.path.join(args.outdir,f"b{args.benchmark}_{args.mode}.pt")
    torch.save({
        "state_dict":best_state,
        "benchmark":args.benchmark,
        "mode":args.mode,
        "size":args.size,
        "best_epoch":best_epoch,
        "val_balanced_accuracy":best_ba,
        "seed":SEED
    },out)
    print("saved:",out)

if __name__=="__main__":
    main()
