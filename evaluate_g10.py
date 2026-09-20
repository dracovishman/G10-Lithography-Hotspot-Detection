import os,glob,argparse,time,json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score

class BasicBlock(nn.Module):
    def __init__(self,in_ch,ch=12):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(in_ch,ch,3),nn.ELU(),
            nn.Conv2d(ch,ch,3),nn.ELU(),nn.Conv2d(ch,ch,3),
            nn.BatchNorm2d(ch),nn.ELU(),nn.MaxPool2d(2))
    def forward(self,x): return self.net(x)

class LightweightCNN(nn.Module):
    def __init__(self,size=64):
        super().__init__(); self.b1=BasicBlock(1); self.b2=BasicBlock(12)
        with torch.no_grad():
            n=self.b2(self.b1(torch.zeros(1,1,size,size))).flatten(1).shape[1]
        self.fc=nn.Sequential(nn.Flatten(),nn.Dropout(.3),nn.Linear(n,10),nn.ELU(),nn.Linear(10,1))
    def forward(self,x): return self.fc(self.b2(self.b1(x))).squeeze(1)

def get_test_files(root,b):
    q=os.path.join(root,f"iccad{b}")
    hs=glob.glob(q+"/test/test_hs/*.png")
    nhs=glob.glob(q+"/test/test_nhs/*.png")
    return hs+nhs,[1]*len(hs)+[0]*len(nhs)

def metrics(y,p):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    sens=tp/(tp+fn) if tp+fn else 0
    spec=tn/(tn+fp) if tn+fp else 0
    return {
      "TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp),
      "balanced_accuracy":(sens+spec)/2,
      "precision":precision_score(y,p,zero_division=0),
      "recall":sens,"specificity":spec,
      "f1":f1_score(y,p,zero_division=0)
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True); ap.add_argument("--benchmark",type=int,required=True)
    ap.add_argument("--checkpoint",required=True); ap.add_argument("--batch",type=int,default=256)
    args=ap.parse_args()
    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    size=ck["size"]; model=LightweightCNN(size); model.load_state_dict(ck["state_dict"]); model.eval()
    paths,labels=get_test_files(args.root,args.benchmark)
    ys=[]; ps=[]; probs=[]; t=time.time()

    with torch.inference_mode():
        for s in range(0,len(paths),args.batch):
            pp=paths[s:s+args.batch]
            arr=np.stack([np.asarray(Image.open(p).convert("L").resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)/255 for p in pp])[:,None]
            z=model(torch.from_numpy(arr))
            pr=torch.sigmoid(z).numpy()
            probs.extend(pr.tolist()); ps.extend((pr>=0.5).astype(int).tolist()); ys.extend(labels[s:s+args.batch])

    r=metrics(ys,ps)
    r.update({"benchmark":args.benchmark,"mode":ck["mode"],
              "best_epoch":ck["best_epoch"],
              "val_balanced_accuracy":ck["val_balanced_accuracy"],
              "test_samples":len(labels),
              "inference_seconds":time.time()-t,
              "inference_ms_per_sample":1000*(time.time()-t)/len(labels)})
    print(json.dumps(r,indent=2))
    out=f"results_b{args.benchmark}_{ck['mode']}.json"
    with open(out,"w") as f: json.dump(r,f,indent=2)
    print("saved:",out)

if __name__=="__main__": main()
