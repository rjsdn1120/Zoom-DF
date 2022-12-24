import os

base = '/media/data2/rjsdn/zoom/dataset/'
val = base+'val'
test = base+'test6'


d = {
    'Face2Face':'F2F',
    'DeepFakeDetection':'DFD',
    'DeepFake':'DF',
    'FaceSwap':'FS',
    'NeuralTextures':'NS'
}
    


m=['R18','X']
f=['5','10','15']
v=[10,30,50,100]

for model in m:
    for k in f:
        for j in v:
            for method in d.keys():
                train = f'/media/data2/rjsdn/zoom/dataset/phase1/train_{j}_{k}/'
                mp = f'weights/{model}_{d[method]}_L2/best.pth'
                q = f'python zoom3.py -bsize 80 -gpu 0 -model {model} -train {train} -val {val} -test {test} -save {model}_{d[method]}_{j}_{k}_L2_T -mp {mp}'
                os.system(q)
