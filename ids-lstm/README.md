
**Project: Data generation and knowledge sharing for robust intrusion detection in IoT systems**

**Funding: Vinnova - Sweden's Inovation Agenncy**

**Authors: Amin Kaveh, Noah Wassberg**

**Institution: Department of Information Technology, Uppsala University, Sweden**

**Dates: April 2022 - October 2024**

**Contact information: christian.rohner@it.uu.se; andreas.johnsson@it.uu.se**

**Manuscript: Amin Kaveh, Noah Wassberg, Christian Rohner, and Andreas Johnsson. "On LSTM Model Generalizability for Intrusion Detection in IoT Networks." NOMS, 2025.**

**Dataset:**  https://ieee-dataport.org/open-access/data-generation-and-knowledge-sharing-robust-intrusion-detection-iot-systems


---------------------------------------------------------------------------------------------------------------------------------

**Step 1** - Use ´obeservableToSink.py´ to preprocess data (mote-output.log file in Cooja's output folder). By running this code, you will have a csv file in the outputfolder, known as features_timeseries_XX_sec.csv. XX is the time interval (binsize in seconds) over which the code calculates the mean and standard deviation of features in UDP packets received by the sink node. In our paper XX = 60 seconds.

**Step 2** - To use a LSTM model we have to decide about the sequence length of input data. As we have multiple timeseries (each Cooja's scenario generates a timeseries data), we can not easily use the sequence length hyperparameter in the LSTM model that we use. Therefore, in the preprocessing phase we generate the sequences and save them in each scenario's folder and then we set the sequence length hyperparameter in the LSTM model as 1. To generate these sequences use ´seqMaker.py´. In our paper we used sequence_length = 10 (10 minutes) as the input.

**Step 3** - To test the generalizability of local models use ´generalizability_[...].py´ files. In these Python codes first the required data is collected fromm all Cooja's scenarios folders (sequence files generated in Step 2) and then we train local LSTM models and test their generalizibilty. ´LSTM_model.py´ and ´aggregate.py´ are used in ´generalizability_[...].py´ codes.

Step 3.1 - To test generalizability in different network topologies (´generalizability_topologies.py´), there should be a file as depth.txt in each Cooja's scenario folder that indicates the maximum depth of the RPL topology in that scenario. To generate these files use ´maxDepth.py´ file.

**Step 4** - To test the generaliàbility of global models trained with the data sharing method use ´dataSharing_[...].py´ files.

**Step 5** - To test the generaliàbility of global models trained with the horizontal federated learning method use ´fedL_[...].py´ files. ´LSTM_FED.py´ is used in this step.

**Step 6** - To evaluate global models' detection accuracy over time use ´dataSharing_[...]_detectionSpeed.py´ and ´fedL_[...]_detectionSpeed.py´.

