from sighan import Dataset

#pku_train = Dataset()
#pku_train.read('training/pku_training.utf8')
#pku_train.output_crfpp_format('pku_train_crf.utf8')

#pku_test = Dataset()
#pku_test.read('gold/pku_test_gold.utf8')
#pku_test.output_crfpp_format('pku_test_crf.utf8')

#pku_result = Dataset()
#pku_result.read_crfpp_result('pku_test_crf_result.txt')
#pku_result.output('pku_segmentation.utf8')

# MSR
#msr_train = Dataset()
#msr_train.read('training/msr_training.utf8')
#msr_train.output_crfpp_format('msr_train_crf.utf8')

#msr_test = Dataset()
#msr_test.read('gold/msr_test_gold.utf8')
#msr_test.output_crfpp_format('msr_test_crf.utf8')

msr_result = Dataset()
msr_result.read_crfpp_result('a.txt')
msr_result.output('msr_segmentation.utf8')
