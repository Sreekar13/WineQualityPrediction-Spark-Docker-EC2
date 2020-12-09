# Wine Quality Prediction Model - Trained in parallel using spark on 4 EC2 instances

**Setting up the cloud environment for training the model in parallel**

-> Create an EMR Spark cluster in AWS with 4 EC2-instances.
**Login into the master instance and do the following,**
-> Install pyspark using ***pip install pyspark***
-> Set environment variables,
***export PYSPARK_PYTHON=/usr/bin/python3
export PYSPARK_DRIVER_PYTHON=/usr/bin/python3***
-> Now, copy wineQualityModel.py, TrainingDataset.csv and ValidationDataset.csv to the master instance and execute the following command to generate the model,

*python3 wineQualityModel.py* 

The above command will generate *target* folder which has the source folder to store the model.

-> Later, install Docker using the below commands,
***sudo yum update -y
sudo amazon-linux-extras install docker
sudo yum install docker
sudo service docker start
sudo usermod -a -G docker hadoop***

-> Now, we have to create the image. So, copy the Dockerfile, target (model) and modelLoad.py into a new directory in master instance and execute the following command to build the image,

***sudo docker build -t winemodel .***

-> Now, we have to tag and push this image to DockerHub using the following command,

***sudo docker tag winemodel sv739/wine-quality-prediction-spark-model
sudo docker push sv739/wine-quality-prediction-spark-model***

P.S:- We are running the python file in container as **ec2-user**. So, the host user should be **ec2-user** while running the image. Intended for security.

-> Now, create a new ec2 instance and login to that new ec2 instance as **ec2-user** and copy the TestDataset.csv into your present working directory in ec2 instance and execute the following command,


***sudo docker run -it -v \`pwd\`/TestDataset.csv:/dataset/TestDataset.csv sv739/wine-quality-prediction-spark-model /dataset/TestDataset.csv***

In the above command, we are volume mapping the TestDataset.csv into the container and running the python file inside the container,

the above command will print the f1 score for the given TestDataset.csv.


Image in DockerHub:- **sv739/wine-quality-prediction-spark-model**
GitHub link for code:- **https://github.com/Sreekar13/WineQualityPrediction-Spark-Docker-EC2**

Files-
[wineQualityModel.py](https://github.com/Sreekar13/WineQualityPrediction-Spark-Docker-EC2/blob/main/wineQualityModel.py)  -- File to generate the model in EMR cluster
[Dockerfile](https://github.com/Sreekar13/WineQualityPrediction-Spark-Docker-EC2/blob/main/Dockerfile) -- File to build image
[modelLoad.py](https://github.com/Sreekar13/WineQualityPrediction-Spark-Docker-EC2/blob/main/modelLoad.py)  -- Python file to load the trained model and compute the F1 Score

