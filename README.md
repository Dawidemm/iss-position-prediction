# iss-position-prediction

## LatLongDataset

### Opis:
Klasa `LatLongDataset` jest opakowaniem wokół danych geolokacyjnych, umożliwiającym ich łatwe przygotowanie do użycia w modelu. Dane są wczytywane z pliku CSV, a następnie przetwarzane w celu usunięcia duplikatów.

### Metody:

#### `__init__(self, csv_file: str, step: int)`
- `csv_file`: Ścieżka do pliku CSV zawierającego dane geolokacyjne.
- `step`: Liczba kroków czasowych między danymi trenującymi a danymi docelowymi.

#### `__len__(self) -> int`
Zwraca liczbę dostępnych przykładów w zestawie danych.

#### `__getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]`
Zwraca krotkę zawierającą dane trenujące i odpowiadające im dane docelowe. Dane są normalizowane i przekształcane w tensory PyTorch.

## LightningLatLongDatamodule

### Opis:
Klasa `LightningLatLongDatamodule` dziedziczy po `LightningDataModule` i służy do łatwego przygotowania danych do treningu, walidacji i testów modelu.

### Metody:

#### `__init__(self, train_csv: str, val_csv: str, test_csv: str, batch_size: int)`
- `train_csv`: Ścieżka do pliku CSV z danymi treningowymi.
- `val_csv`: Ścieżka do pliku CSV z danymi walidacyjnymi.
- `test_csv`: Ścieżka do pliku CSV z danymi testowymi.
- `batch_size`: Rozmiar batcha używanego w DataLoaderach.

#### `setup(self, stage: str) -> None`
Inicjalizuje zestawy danych treningowych, walidacyjnych i testowych.

#### `train_dataloader(self) -> DataLoader`
Zwraca DataLoader dla danych treningowych.

#### `val_dataloader(self) -> DataLoader`
Zwraca DataLoader dla danych walidacyjnych.

#### `test_dataloader(self) -> DataLoader`
Zwraca DataLoader dla danych testowych.
