import sys, getopt
import os


def get_config_type(config_file):
    return os.path.splitext(os.path.basename(config_file))[0].split('_')[1]

def get_param(line):
    return line.split(':')[1].split('#')[0].strip()

def args_parsing(config_file):
    if get_config_type(config_file) == 'preprocessing':
        f = open(config_file, 'r')
        root, raw_data_path, preprocessed_data_path = [get_param(line) for line in f]
        f.close()
        
        raw_data_path = os.path.join(root, raw_data_path)
        preprocessed_data_path = os.path.join(root, preprocessed_data_path)
        
        params = {'root': root,
                  'raw_data_path': raw_data_path,
                  'preprocessed_data_path': preprocessed_data_path}
        
        return params
    
    if get_config_type(config_file) == 'train':
        f = open(config_file, 'r')
        root, experiment_name, image_size, batch_size, lr, n_epochs, log_dir, checkpoint_path = [get_param(line) for line in f]
        
        image_size = tuple(map(int, image_size.split(", ")))
        batch_size = int(batch_size)
        lr = float(lr)
        n_epochs = int(n_epochs)
        log_dir = os.path.join(root, log_dir)
        if experiment_name == 'None':
            experiment_name = None
        if checkpoint_path != 'None':
            checkpoint_path = os.path.join(root, checkpoint_path)
        else:
            checkpoint_path = None
        
        f.close()
        
        params = {'root': root,
                  'experiment_name': experiment_name,
                  'image_size': image_size,
                  'batch_size': batch_size,
                  'lr': lr,
                  'n_epochs': n_epochs,
                  'log_dir': log_dir,
                  'checkpoint_path': checkpoint_path
                  }
        
        return params
    
    if get_config_type(config_file) == 'inference':
        f = open(config_file, 'r')
        root, model_path, image_size, test_data_path = [get_param(line) for line in f]
        f.close()
        
        model_path = os.path.join(root, model_path)
        image_size = tuple(map(int, image_size.split(", ")))
        test_data_path = os.path.join(root, test_data_path)
        
        params = {'model_path': model_path,
                  'image_size': image_size,
                  'test_data_path': test_data_path}
        
        return params

def cmd_args_parsing(argv):
    try:
        opts, args = getopt.getopt(argv, "h", ["config_file="])
    except getopt.GetoptError:
        print("args: --config_file=<path to config file>")
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print("args: --config_file=<path to config file>")
            sys.exit()
        elif opt == "--config_file":
            config_file = arg
            
    return config_file
