import torch
import cv2
import numpy as np

# timeSformerのモデルをダウンロード
model = torch.hub.load('facebookresearch/TimeSformer', 'TimeSformer_divST_8x32_224', pretrained=True)
# カメラからの映像を取得
capture = cv2.VideoCapture(0)

def preprocess(video):
    # 動画を8フレームごとに分割
    video = video[::8]
    # 各フレームを224×224にリサイズ
    video = np.array([cv2.resize(frame, (224, 224)) for frame in video])
    # RGBからBGRに変換
    video = video[:, :, :, ::-1]
    # 正規化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    video = (video / 255.0 - mean) / std
    # バッチ次元とチャンネル次元を追加
    video = np.expand_dims(video, axis=0)
    video = np.transpose(video, (0, 4, 1, 2, 3))
    # テンソルに変換
    video = torch.from_numpy(video).float()
    return video

    # モデルを評価モードに設定
    model.eval()
    # カテゴリの名前を読み込む
    categories = np.loadtxt('categories.txt', dtype=str)

while True:
    # カメラから1秒間の映像を取得
    ret, video = capture.read()
    # 映像をtimeSformerに入力できる形式に変換
    video = preprocess(video)
    # モデルに映像を入力して確率分布を出力
    with torch.no_grad():
        output = model(video)
    # 確率分布から最も高い確率を持つカテゴリのインデックスを取得
    pred = output.argmax(dim=1).item()
    # カテゴリのインデックスからカテゴリの名前を取得
    # name = categories[pred]
    name = pred
    # 判断結果をカメラの映像に重ねる
    cv2.putText(video, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # カメラの映像を表示
    cv2.imshow('camera', video)
    # qキーを押すと終了