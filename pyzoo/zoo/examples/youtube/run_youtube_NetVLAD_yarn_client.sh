export ANALYTICS_ZOO_HOME=/root/jwang/zoo_dist_youtube/dist
export SPARK_HOME=/opt/work/spark-2.0.0-bin-hadoop2.7/
export VENV_HOME=/root/jwang/zoo_dist_youtube/dist/bin
export JAVA_HOME=/opt/jdk
export HADOOP_HDFS_HOME=/opt/work/hadoop-2.7.2
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${JAVA_HOME}/jre/lib/amd64/server

CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob) \
PYSPARK_DRIVER_PYTHON=${VENV_HOME}/venv/bin/python \
PYSPARK_PYTHON=venv.zip/venv/bin/python \
$ANALYTICS_ZOO_HOME/bin/spark-submit-with-zoo.sh \
 --master yarn \
 --deploy-mode client \
 --executor-memory 150g \
 --driver-memory 150g \
 --executor-cores 2 \
 --num-executors 2 \
 --archives ${VENV_HOME}/venv.zip \
 --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" \
 --conf "spark.executor.extraLibraryPath=/opt/jdk/jre/lib/amd64/server" \
 --conf "spark.executorEnv.LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${JAVA_HOME}/jre/lib/amd64/server" \
 --conf "spark.executorEnv.CLASSPATH=$CLASSPATH:$(${HADOOP_HOME}/bin/hadoop classpath --glob)" \
 --conf "spark.driver.maxResultSize=10g" \
 --jars ${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_0.8.0-spark_2.1.0-0.6.0-SNAPSHOT-jar-with-dependencies.jar,/root/jwang/tensorflow-hadoop-1.10.0.jar \
 --py-files /root/jwang/Youtube-8M-WILLOW.zip,${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_0.8.0-spark_2.1.0-0.6.0-SNAPSHOT-python-api.zip \
 /root/jwang/zoo_dist_youtube/pyzoo/zoo/examples/youtube/train_youtube_2.py \
 --train_data_pattern=hdfs://almaren-node-003:9000/youtube/frame/val_small \
 --model=NetVLADModelLF \
 --train_dir=/root/jwang/model/youtube/frame/sample_model \
 --frame_features=True \
 --feature_names="rgb,audio" \
 --feature_sizes="1024,128" \
 --batch_size=64 \
 --base_learning_rate=0.0002 \
 --netvlad_cluster_size=32 \
 --netvlad_hidden_size=1024 \
 --moe_l2=1e-6 \
 --iterations=300 \
 --learning_rate_decay=0.8 \
 --netvlad_relu=False \
 --gating=True \
 --moe_prob_gating=True \
 --num_epochs=10 \
 --netvlad_add_batch_norm=False

