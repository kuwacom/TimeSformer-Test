# テストの前に以下を実行してください
# mim download mmaction2 --config tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb --dest ./mmaction2

from operator import itemgetter
from mmaction.apis import init_recognizer, inference_recognizer

config_file = './mmaction2/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py'
checkpoint_file = './mmaction2/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth'
video_file = './mmaction2/demo/demo.mp4'
label_file = './mmaction2/tools/data/kinetics/label_map_k400.txt'
model = init_recognizer(config_file, checkpoint_file, device='cuda:0')  # or device='cpu'
pred_result = inference_recognizer(model, video_file)

pred_scores = pred_result.pred_score.tolist()
score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
top5_label = score_sorted[:5]

labels = open(label_file).readlines()
labels = [x.strip() for x in labels]
results = [(labels[k[0]], k[1]) for k in top5_label]

print('The top-5 labels with corresponding scores are:')
for result in results:
    print(f'{result[0]}: ', result[1])