export ANALYTICS_ZOO_HOME=/home/jwang/git/analytics-zoo-jennie4/dist
export SPARK_HOME=/tools/spark-2.2.0-bin-hadoop2.7/

$ANALYTICS_ZOO_HOME/bin/spark-submit-with-zoo.sh \
 --master local[4] \
 --driver-memory 80g \
 --jars /home/jwang/git/ecosystem/hadoop/target/tensorflow-hadoop-1.10.0.jar \
 --py-files /home/jwang/git/Youtube-8M-WILLOW/Youtube-8M-WILLOW.zip \
 /home/jwang/git/analytics-zoo-jennie4/pyzoo/zoo/examples/youtube/train_youtube.py \
 --train_data_pattern=/data/youtube/frame/val_small/*.tfrecord \
 --model=NetVLADModelLF \
 --train_dir=/model/youtube/frame/sample_model \
 --frame_features=True \
 --feature_names="rgb,audio" \
 --feature_sizes="1024,128" \
 --batch_size=16 \
 --base_learning_rate=0.0002 \
 --netvlad_cluster_size=16 \
 --netvlad_hidden_size=128 \
 --moe_l2=1e-6 \
 --iterations=300 \
 --learning_rate_decay=0.8 \
 --netvlad_relu=False \
 --gating=True \
 --moe_prob_gating=True \
 --num_epochs=1 \
 --netvlad_add_batch_norm=False
