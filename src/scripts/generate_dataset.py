from src.modules.dataset_generator import GenerateDataset

def generate_dataset(type: str, samples: int):
    GenerateDataset(type=type, samples=samples).save_as_csv()

if __name__ == '__main__':
    generate_dataset(type='new_train', samples=700)
    generate_dataset(type='new_val', samples=200)
    generate_dataset(type='new_test', samples=100)