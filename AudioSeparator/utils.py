import os
import csv
import wget
import tarfile


#########################################
#///////////////////////////////////////#
#########################################


def buildFileTree():
    # Make dataset
    if not os.path.exists('datasets'):
        os.makedirs('datasets')

        if not os.path.exists('datasets/sample_data'):
            os.makedirs('datasets/sample_data')

        if not os.path.exists('datasets/fuss'):
            os.makedirs('datasets/fuss')
            if not os.path.exists('datasets/fuss/tf_train'):
                os.makedirs('datasets/fuss/tf_train')
            if not os.path.exists('datasets/fuss/tf_val'):
                os.makedirs('datasets/fuss/tf_val')
            if not os.path.exists('datasets/fuss/tf_eval'):
                os.makedirs('datasets/fuss/tf_eval')
            if not os.path.exists('datasets/fuss/tf_train_dm'):
                os.makedirs('datasets/fuss/tf_train_dm')

        if not os.path.exists('datasets/librimix'):
            os.makedirs('datasets/librimix')
            if not os.path.exists('datasets/librimix/tf_train'):
                os.makedirs('datasets/librimix/tf_train')
            if not os.path.exists('datasets/librimix/tf_val'):
                os.makedirs('datasets/librimix/tf_val')
            if not os.path.exists('datasets/librimix/tf_eval'):
                os.makedirs('datasets/fuss/tf_eval')
            if not os.path.exists('datasets/librimix/tf_train_dm'):
                os.makedirs('datasets/fuss/tf_train_dm')

    # Make configs
    if not os.path.exists('configs'):
        os.makedirs('configs')

    # Make log
    if not os.path.exists('log'):
        os.makedirs('log')

    # Make model weights mix
    if not os.path.exists('model_weights'):
        os.makedirs('model_weights')


#########################################
#///////////////////////////////////////#
#########################################


def downloadFuss():
    ''' downloads reverb version '''
    fuss_url = "https://zenodo.org/record/3694384/files/FUSS_ssdata_reverb.tar.gz?download=1"
    wget.download(fuss_url, 'datasets/fuss.tar.gz')
    tar = tarfile.open('datasets/fuss.tar.gz', "r:gz")
    tar.extractall('datasets/fuss')
    tar.close()


#########################################
#///////////////////////////////////////#
#########################################


def checkExperimentBuilt(EXPERIMENT_NAME):
    if os.path.isdir('model_weights/'+EXPERIMENT_NAME) == False:
        return False
    if os.path.isdir('log/'+EXPERIMENT_NAME) == False:
        return False
    return True

def buildExperiment(EXPERIMENT_NAME,configs):
    # Make dir in model weights named EXPERIMENT_NAME
    if os.path.isdir('model_weights/'+EXPERIMENT_NAME) != True:
        os.mkdir('model_weights/'+EXPERIMENT_NAME)

    # Make folder in log called EXPERIMENT_NAME with two csv files EXPERIMENT_NAME_train.py and EXPERIMENT_NAME_eval.csv
    if os.path.isdir('log/'+EXPERIMENT_NAME) != True:
        os.mkdir('log/'+EXPERIMENT_NAME)
        with open('log/'+EXPERIMENT_NAME+'/'+EXPERIMENT_NAME +'_train.csv', mode='w') as train_log:
            train_log = csv.writer(train_log, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            train_log.writerow(['n_epoch','t_snr_vary_n_source','t_si_snri','t_sdri','v_snr_vary_n_source','v_si_snri','v_sdri','gpu','batch_size'])
        with open('log/'+EXPERIMENT_NAME+'/'+EXPERIMENT_NAME +'_result.csv', mode='w') as result_log:
            result_log = csv.writer(result_log, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            result_log.writerow(['n_epoch','e_snr_vary_n_source','e_si_snri','e_sdri','gpu','batch_size'])

def updateTrainLog(EXPERIMENT_NAME,configs,history):
    with open('log/'+EXPERIMENT_NAME+'/'+EXPERIMENT_NAME +'_train.csv', 'a', newline='') as train_log:
        train_log = csv.writer(train_log, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        vals = [configs['epochs'],history.history['snr_vary_n_source'][-1],history.history['si-snri'][-1],\
                    history.history['sdri'][-1],history.history['val_snr_vary_n_source'][-1],\
                    history.history['val_si-snri'][-1],history.history['val_sdri'][-1],configs['gpu'],configs['batch_size']]
        train_log.writerow(vals)

def updateResultLog(EXPERIMENT_NAME,configs,results):
    with open('log/'+EXPERIMENT_NAME+'/'+EXPERIMENT_NAME +'_result.csv', 'a', newline='') as result_log:
        result_log = csv.writer(result_log, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        vals = [configs['epochs'],results['snr_vary_n_source'].numpy(),results['si-snri'].numpy(),\
                    results['sdri'].numpy(),configs['gpu'],configs['batch_size']]
        result_log.writerow(vals)

def makeReport(EXPERIMENT_NAME):
    pass

def cleanExperiment():
    pass
