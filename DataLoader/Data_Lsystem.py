from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='fractal_random',
                       captions_per_image=1,
                       min_word_freq=1,
                       output_folder="Data_Generation/L_system/100padding/",
                       data_path="Data_Generation/L_system/Data/",
                       max_len=100)
