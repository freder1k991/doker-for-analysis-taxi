#войти в папку cd
#bash start_local_cluster.sh

#docker stop
#wsl.exe --list --online - дистрибутивы
#wsl.exe --set-default-version 2 - указание версию по умолчанию 
# wsl.exe --status - проверка статуса 
#!/bin/bash

# Full path to project directory
# We mount it to docker containers
# So they will se the content in that directory
PATH_TO_PROJECT_DIR="/mnt/d/BigData"

MEMORY_PER_WORKER='2g'
CORES_PER_WORKER=1

# Creates local docker network and names it as "spark_network"
docker network create spark_network

# Runs Spark Master Node
docker run -d -p 8080:8080 -p 7077:7077 --name spark_master --network spark_network \
-v $PATH_TO_PROJECT_DIR:/work:rw \
apache/spark:latest /opt/spark/bin/spark-class org.apache.spark.deploy.master.Master \
-h spark_master

# Save IP address of our Spark Master node 
# to attach workers to it
SPARK_MASTER_IP=`docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' spark_master`

# Run Spark workers and bind them to our Spark Master node
docker run -d --name spark_worker1 --network spark_network \
-e SPARK_WORKER_MEMORY=$MEMORY_PER_WORKER \
-e SPARK_WORKER_CORES=$CORES_PER_WORKER \
-v $PATH_TO_PROJECT_DIR:/work:rw \
apache/spark:latest /opt/spark/bin/spark-class org.apache.spark.deploy.worker.Worker \
spark://$SPARK_MASTER_IP:7077 

docker run -d --name spark_worker2 --network spark_network \
-e SPARK_WORKER_MEMORY=$MEMORY_PER_WORKER \
-e SPARK_WORKER_CORES=$CORES_PER_WORKER \
-v $PATH_TO_PROJECT_DIR:/work:rw \
apache/spark:latest /opt/spark/bin/spark-class org.apache.spark.deploy.worker.Worker \
spark://$SPARK_MASTER_IP:7077 

docker run -d --name spark_worker3 --network spark_network \
-e SPARK_WORKER_MEMORY=$MEMORY_PER_WORKER \
-e SPARK_WORKER_CORES=$CORES_PER_WORKER \
-v $PATH_TO_PROJECT_DIR:/work:rw \
apache/spark:latest /opt/spark/bin/spark-class org.apache.spark.deploy.worker.Worker \
spark://$SPARK_MASTER_IP:7077 

docker run -d --name spark_worker4 --network spark_network \
-e SPARK_WORKER_MEMORY=$MEMORY_PER_WORKER \
-e SPARK_WORKER_CORES=$CORES_PER_WORKER \
-v $PATH_TO_PROJECT_DIR:/work:rw \
apache/spark:latest /opt/spark/bin/spark-class org.apache.spark.deploy.worker.Worker \
spark://$SPARK_MASTER_IP:7077 


docker run -d --name jupyter_lab -p 10000:8888 -p 4040:4040 --network spark_network \
--hostname localhost \
--user root \
-e GRANT_SUDO=yes \
-v $PATH_TO_PROJECT_DIR:/work:rw \
-e SPARK_MASTER=spark://$SPARK_MASTER_IP:7077 \
jupyter/pyspark-notebook start-notebook.sh --NotebookApp.token='' --NotebookApp.notebook_dir='/work'


echo 'YOUR SPARK MASTER NODE IP IS:' $SPARK_MASTER_IP
echo 'YOU CAN ACCESS JUPYTER LAB VIA: http://localhost:10000'