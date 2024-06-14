# テストの前に以下を実行してください
# mim download mmaction2 --config tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb --dest ./mmaction2

import os
import torch
import cv2
from mmaction.apis import inference_recognizer, init_recognizer

def resize_frame(frame, width=None, height=None):
    """
    フレームのサイズを変更する関数。

    Args:
        frame (numpy.ndarray): サイズを変更するフレーム（画像）。
        width (int, optional): 変更後の幅。デフォルトは None。
        height (int, optional): 変更後の高さ。デフォルトは None。

    Returns:
        numpy.ndarray: サイズを変更したフレーム。
    """
    if width is None and height is None:
        return frame

    if width is None:
        aspect_ratio = height / float(frame.shape[0])
        width = int(frame.shape[1] * aspect_ratio)
    elif height is None:
        aspect_ratio = width / float(frame.shape[1])
        height = int(frame.shape[0] * aspect_ratio)

    resized_frame = cv2.resize(frame, (width, height))

    return resized_frame

def get_resized_dimensions(frame, width=None, height=None):
    """
    フレームをリサイズした後のサイズを取得する関数。

    Args:
        frame (numpy.ndarray): サイズを変更するフレーム（画像）。
        width (int, optional): 変更後の幅。デフォルトは None。
        height (int, optional): 変更後の高さ。デフォルトは None。

    Returns:
        tuple: 変更後の幅と高さのタプル。
    """
    if width is None and height is None:
        return frame.shape[1], frame.shape[0]

    if width is None:
        aspect_ratio = height / float(frame.shape[0])
        width = int(frame.shape[1] * aspect_ratio)
    elif height is None:
        aspect_ratio = width / float(frame.shape[1])
        height = int(frame.shape[0] * aspect_ratio)

    return width, height


def insert_text_on_frame(frame, text, org=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.4,
                         text_color=(255, 255, 255), outline_color=(0, 0, 0), outline_size=2, thickness=2, line_spacing=30):
    """
    フレームにテキストを挿入し、黒でアウトラインを付ける

    Args:
        frame (numpy.ndarray): テキストを挿入するフレーム（画像）
        text (str): 挿入するテキスト
        org (tuple, optional): テキストを挿入する位置 (x, y)。デフォルトは (10, 30)
        font (int, optional): 使用するフォント。デフォルトは cv2.FONT_HERSHEY_SIMPLEX
        font_scale (float, optional): フォントのスケール。デフォルトは 0.4
        text_color (tuple, optional): テキストの色 (B, G, R)。デフォルトは (255, 255, 255)
        outline_color (tuple, optional): アウトラインの色 (B, G, R)。デフォルトは (0, 0, 0)
        thickness (int, optional): テキストとアウトラインの太さ。デフォルトは 2
        line_spacing (int, optional): 行間のスペース。デフォルトは 30

    Returns:
        numpy.ndarray: テキストを挿入したフレーム
    """
    # テキストを改行で分割
    lines = text.split('\n')

    # 改行ごとにテキストを挿入
    y = org[1]
    for line in lines:
        # アウトラインを描画
        cv2.putText(frame, line, (org[0], y), font, font_scale, outline_color, thickness + outline_size, cv2.LINE_AA)
        # テキストを描画
        cv2.putText(frame, line, (org[0], y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        y += int(font_scale * line_spacing)

    return frame

def split_segment_detect(video_path, model, label_path, test_pipeline=None, processing_frame_height=None, segment_duration=10, output_video_path='./output_video.mp4', temp_path='./temp'):
    """
    動画をセグメントに分割し、各セグメントを推論モデルで処理して結果を描画し、結合して出力

    Args:
        video_path (str): 入力動画ファイルのパス
        model (nn.Module): 推論に使用するモデル
        label_path (str): ラベルマップのファイルパス
        test_pipeline (Optional[Compose]): テスト用の前処理パイプライン。デフォルトはNone
        processing_frame_height (Optional[int]): 処理時のフレームの高さ。デフォルトはNone
        segment_duration (int): セグメントの長さ（秒単位）。デフォルトは10
        output_video_path (str): 出力動画のパス。デフォルトは'./output_video.mp4'
        temp_path (str): 一時ファイルを保存するディレクトリのパス。デフォルトは'./temp'

    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if (processing_frame_height == None):
        processing_frame_height = frame_height

    # segment_duration 秒間のフレーム数を計算
    frames_per_segment = int(fps * segment_duration)
    print(f'Total Frame: {str(total_frames)}')

    os.makedirs(temp_path, exist_ok=True)
    
    processed_videos = []
    frame_num = 0
    while frame_num < total_frames:
        # セグメントの最初のフレーム番号を設定
        start_frame = frame_num
        end_frame = min(start_frame + frames_per_segment, total_frames)  # セグメントの終了フレーム番号
        print(f'NOW Processing Frame:{str(start_frame)} ~ {str(end_frame)}')

        # セグメントのフレームを一時ファイルとして保存
        segment_frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            segment_frames.append(frame)
            frame_num += 1

        # セグメントを一時ファイルに書き出す
        segment_filename = f'{temp_path}/segment_{start_frame}_{end_frame}.mp4'
        out = cv2.VideoWriter(segment_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, get_resized_dimensions(frame, None, processing_frame_height))
        for frame in segment_frames:
            out.write(resize_frame(frame, None, processing_frame_height))
        out.release()

        # 推論結果を取得してテキストを挿入した動画を生成
        # data = dict(filename=segment_filename, label=-1, start_index=start_frame + idx, modality='RGB')
        pred_result = inference_recognizer(model, segment_filename, test_pipeline)

        # 推論結果を整形
        pred_scores = pred_result.pred_score.tolist()
        score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
        score_sorted = sorted(score_tuples, key=lambda x: x[1], reverse=True)
        top5_label = score_sorted[:5]

        labels = open(label_path).readlines()
        labels = [x.strip() for x in labels]
        results = [(labels[k[0]], k[1]) for k in top5_label]

        text_added = []
        for idx, frame in enumerate(segment_frames):
            text_to_display = 'kuwacom/TimeSformer-Tools\n'
            text_to_display += f'Segment Interval: {segment_duration}s\n'
            text_to_display += f'Segment Frame: {start_frame} ~ {end_frame}\n'  # セグメントの範囲を表示
            text_to_display += f'Frame: {start_frame + idx}/{total_frames}\n'
            
            # リザルトを改行していい感じにする
            for idx, result in enumerate(results):
                text_to_display += f'{idx+1}. {result[0]}: {result[1]:.3f}\n'
            
            # フレームにテキストを挿入
            frame_with_text = insert_text_on_frame(frame.copy(), text_to_display, font_scale=(frame_height / 1000))
            text_added.append(frame_with_text)
            
        cv2.imshow('Current Processing Frame', text_added[0])
        cv2.waitKey(1)
        # 処理済み動画を保存
        processed_videos.append(text_added)

    cap.release()

    # 処理済み動画を結合して出力
    save_video(output_video_path, concat_videos(processed_videos), fps, (frame_width, frame_height))
    print(f'処理済みの動画を {output_video_path} に書き出しました。')

def concat_videos(video_segments):
    return [item for sublist in video_segments for item in sublist]
    
def save_video(filename, frames, fps, frame_size):
    frame_width, frame_height = frame_size
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    for frame in frames:
        out.write(frame)
    out.release()

# MMAction2の設定ファイルとチェックポイントファイル
config_file = './mmaction2/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py'
checkpoint_file = './mmaction2/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth'

# MMAction2のモデルを初期化
model = init_recognizer(config_file, checkpoint_file, device='cuda:0')
if torch.cuda.is_available():
    model.half() # fp16

# 処理する動画ファイルとラベルファイルのパス
video_path = './mmaction2/demo/demo.mp4'
label_path = './mmaction2/tools/data/kinetics/label_map_k400.txt'

output_video_path = './output.mp4'

# 動画の処理とプレビューの表示を実行
split_segment_detect(video_path, model, label_path, segment_duration=2, output_video_path=output_video_path)
