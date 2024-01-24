# timeSformer vnev version
# 動作チェックにどうぞ
import torch
import cv2
from timesformer.models.vit import TimeSformer
import matplotlib.pyplot as plt

from operator import itemgetter

# OpenCVを使用して動画を読み込む
cap = cv2.VideoCapture("./dataset/test/00065.mp4")


current_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
current_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# フレームの次元（高さ、幅）を指定
height = 200
width = int((current_width / current_height) * height)

# frames = int(cap.get(cv2.CAP_PROP_frames_COUNT))
frames = 10

# モデルの入力となるテンソルを初期化
video_tensor = torch.empty(2, 3, frames, height, width)

# 動画の各フレームに対して処理を行う
for frames_idx in range(frames):
    ret, frame = cap.read()

    # # サイズをモデルに合わせて変更
    frame = cv2.resize(frame, (width, height))
    
    # チャンネルの次元を追加し、PyTorchのテンソルに変換
    frames_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()

    # テンソルにフレームを代入
    video_tensor[:, :, frames_idx, :, :] = frames_tensor

    # 描画したフレームを表示
    cv2.imshow('Video', frame)

    # 'q'キーが押されたら終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()


print("Loading Model...")
# モデルを初期化
model = TimeSformer(
    img_size=int(height),
    num_classes=400,
    num_frames=frames,
    attention_type='divided_space_time', 
    pretrained_model='./models/TimeSformer_divST_8x32_224_K400.pyth'
)
print("Done")

print("Detect Video...")
pred = model(video_tensor, )
print("Done")

print(pred.shape)
print(type(pred))