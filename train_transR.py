import config
import models
import tensorflow as tf
import numpy as np

# Train TransR based on pretrained TransE results.
# ++++++++++++++TransNB++++++++++++++++++++

con = config.Config()
# con.set_in_path("./benchmarks/train_data/")
con.set_train_times(100)
con.set_sample_node_number(50)
con.set_sample_node_neighbour(20)
con.set_alpha(0.5)
con.set_margin(5.0)
con.set_rel_type(2)
con.set_dimension(100)
con.set_ent_neg_rate(10)
con.set_opt_method("SGD")
con.set_export_files("./res/modelE_even5.vec.tf", 0)
# Model parameters will be exported to json files automatically.
con.set_out_files("./res/embeddingE_even5.vec.json")
# con.set_import_files("./res/model1.vec.tf")
con.init()

con.set_model(models.TransE)
parameters = con.get_parameters("numpy")
parameters["type_transfer_matrix"] = np.array(
    [(np.identity(100).reshape((100*100))) for i in range(con.get_rel_type())])
con.set_parameters(parameters)
con.run()

parameters = con.get_parameters("numpy")
# ++++++++++++++TransR++++++++++++++++++++
conR = config.Config()
# Input training files from benchmarks/FB15K/ folder.
# conR.set_in_path("./benchmarks/train_data/")
# True: Input test files from the same folder.
# conR.set_test_link_prediction(True)

conR.set_work_threads(8)
conR.set_train_times(100)
conR.set_sample_node_number(50)
conR.set_sample_node_neighbour(20)
conR.set_alpha(1.0)
conR.set_margin(5.0)
conR.set_bern(1)
conR.set_dimension(100)
conR.set_ent_neg_rate(1)
conR.set_rel_neg_rate(0)
conR.set_opt_method("SGD")

# Models will be exported via tf.Saver() automatically.
conR.set_export_files("./res/modelR_even2.vec.tf", 0)
# Model parameters will be exported to json files automatically.
conR.set_out_files("./res/embeddingR_even2.vec.json")
# Initialize experimental settings.
conR.init()
# Load pretrained TransE results.
conR.set_model(models.TransR)
parameters["transfer_matrix"] = np.array(
    [(np.identity(100).reshape((100*100))) for i in range(conR.get_rel_total())])
conR.set_parameters(parameters)
# Train the model.
conR.run()
# To test models after training needs "set_test_flag(True)".
