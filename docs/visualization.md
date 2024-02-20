### - visualization.py

This script provides a real-time visualization of the predictions made by the trained model (`LightningLatLongPredictor`) on longitude and latitude coordinates. The visualization includes both true and predicted points in a 3D plot.

#### Usage

1. **Navigate to the `is-position-prediction` directory** in your terminal.
2. Run the script:

    ```bash
    python src/visualization.py
    ```

#### Real-time Visualization

The `visualization.py` script does the following:

1. Loads a pre-trained PyTorch Lightning model (`LightningLatLongPredictor`) from a specified checkpoint using the `get_model_checkpoint_path` utility function.

2. Sets up a 3D plot using Matplotlib for visualizing true and predicted longitude and latitude coordinates in real-time.

3. Defines an update function (`update_plot`) for the Matplotlib animation, which:
   - Fetches real-time data using the `FetchData` class.
   - Processes the data for input to the model.
   - Makes predictions using the pre-trained model.
   - Updates the 3D plot with true and predicted points.
   - Calculates and displays Mean Absolute Error (MAE) between true and predicted coordinates.

4. Initializes a Matplotlib FuncAnimation object (`ani`) to continuously update the plot in real-time.

5. Allows pausing and resuming the animation by pressing any key.

Note: Users can change the utilized model by providing their model checkpoint path. This can be achieved by modifying the `get_model_checkpoint_path` function in the script by setting argument to `selection='last'`, which returns the checkpoint path for the `load_from_checkpoint` method.