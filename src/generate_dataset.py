from modules.dataset_generator import GenerateDataset

def generate_dataset(type: str, duration: int):
    GenerateDataset(type=type, duration=duration).save_as_csv()

if __name__ == '__main__':
    generate_dataset(type='new_train', duration=700)
    generate_dataset(type='new_val', duration=200)
    generate_dataset(type='new_test', duration=100)