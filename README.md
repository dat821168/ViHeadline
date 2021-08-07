<h1 align="center"> Vi-Headline Gereration </h1>
<h4 align="center">
    <p>
        <b>Tiếng Việt</b> |
        <a href="">English</a>
    <p>
</h4>

<h3 align="center">
    <p>Mô hình tóm tắt trích yếu cho bài toán sinh tiêu đề tin tức tiếng Việt tự động</p>
</h3>

ViHeadline áp dụng mô hình [<b>BERTSUM</b>](https://arxiv.org/abs/1908.08345) cho bài toàn sinh tiêu tự động theo hướng tiếp cập tóm tắt trích yếu (Abstractive Summarization). Mô hình tận dụng sức mạnh các mô hình ngôn ngữ BERT và các biến thể của nó đã được huấn luyện trước trên các tập dữ liệu lớn, kết hợp một số kỹ thuật tinh chỉnh kiến trúc giúp mở rộng khả năng biểu diễn văn bản với nội dung chứa nhiều cầu. Chi tiết tham khảo tại bài [<b>Text Summarization with Pretrained Encoders</b>](https://arxiv.org/abs/1908.08345)

Code được xây dựng dựa trên nguồn <b>PreSumm</b>(https://github.com/nlpyang/PreSumm) và <b>OpenNMT</b>(https://github.com/OpenNMT/OpenNMT-py)

## Installation
Requirements:
- Python 3.7+
- Pytorch 1.4+
- Transformers 4.8.2+
- TensorboardX 2.4+
- VnCoreNLP (Nếu sử dụng PhoBERT cho encoder)
- Pyrouge 0.1.3+ (Có thể tham khảo việc install tại https://github.com/bheinzerling/pyrouge và https://github.com/andersjo/pyrouge)

Install dependencies:
`
pip install -r requirements.txt
`
## Dataset format
Mỗi bài tin tức nên được lưu trữ trong một file json với format

```yaml
{
  "headline": "Nội dung tiêu đề",
  "contain": "Nội dun bài viết"
}
```

Ví dụ:

```yaml
{
  "headline": "Lịch thi đấu Việt Nam tại Asian Cup 2019",
  "content": "Tại giải đấu năm nay, ngoài hai đội nhất nhì mỗi bảng còn có 4 đội hạng ba có thành tích tốt nhất giành quyền vào vòng 16 đội. Do đó, tuyển Việt Nam có cơ hội vượt qua vòng bảng nếu cầm hòa Iraq và đánh bại Yemen."
}
```

Dữ liệu được sử dụng cho mô hình pretrain được thu thập từ một trang tin tức phổ biến ở Việt Nam [<b>Tuổi Trẻ</b>](https://tuoitre.vn). Tổng gồm 127.439 bài viết được chia thành 3 tập train/validate/test theo tỉ lệ tương ứng  0,8/0,1/0,1.

## Run the code
### Dataset Cleanning
```
python viheadline/preprocess.py -mode build -pretrained_model phobert -raw_path ./data/raw_data/TuoiTreNews  -save_path ./data/clean_data -n_cpus 4 -log_file ./logs/tuoitre_build.log
```

### Dataset Building
```
python viheadline/preprocess.py -mode build -pretrained_model phobert -raw_path ./data/clean_data -map_path ./data/map_data -save_path ./data/bert_data -segment_path ./resources/VnCoreNLP-master/VnCoreNLP-1.1.1.jar -n_cpus 4 -shard_size 10000 -log_file ./logs/tuoitre_build.log
```

### Model Training
```
python train.py -mode train -encoder phobert -bert_data ./data/bert_data_tmp/ -save_model ./models/ -batch_size 140 -max_pos 256 -lr_bert 2e-3 -lr_dec 0.1 -share_emb -finetune_bert -dec_dropout 0.2 -save_checkpoint_steps 2500 -accum_count 5 -report_every 2500 -train_steps 200000 -visible_gpus 0 -use_interval true -log_file ./logs/tuoitre_train.log
```

### Model Evaluation
```
 python train.py -mode validate -encoder phobert -batch_size 140 -test_batch_size 140 -bert_data ./data/bert_data_tmp/ -save_model ./models/ -sep_optim true -use_interval true -visible_gpus 0 -max_pos 256 -max_length 200 -alpha 0.95 -min_length 50 -result_path ./results/ -log_file ./logs/tuoitre_train.log
 ```