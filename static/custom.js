document.addEventListener('DOMContentLoaded', function () {
    
    const analysisTypeButtons = document.querySelectorAll('.analysis-type-btn');
    const chartTypeButtonsContainer = document.getElementById('chart-type-buttons-container');
    const hiddenInput = document.getElementById('selected-chart-type');
    const visualizationResultsContainer = document.getElementById('visualizationResults');

    analysisTypeButtons.forEach(button => {
        button.addEventListener('click', function () {
            // Set the hidden input value to the clicked button's value
            hiddenInput.value = this.value;

            // Show/hide chart type buttons based on analysis type
            updateChartTypeButtons();
        });
    });
    var datasetDropdown = document.getElementById("dataset-dropdown");
    var fileInput = document.querySelector(".chooseFile");
    var hiddenDatasetInput = document.getElementById("hidden-dataset");

    datasetDropdown.addEventListener("change", function() {
        var selectedDataset = datasetDropdown.value;
        if (selectedDataset) {
            fileInput.setAttribute("data-dataset", selectedDataset);
            hiddenDatasetInput.value = selectedDataset;
        } else {
            fileInput.removeAttribute("data-dataset");
            hiddenDatasetInput.value = "";
        }
    });

    function updateChartTypeButtons() {
        // Clear existing chart type buttons
        chartTypeButtonsContainer.innerHTML = '';

        const chartButtonsMap = {
            univariate: ['histogram', 'box_plot', 'violin_plot', 'dot_plot', 'quantile-quantile_plot', 'density_plot', 'distribution_plot'],
            bivariate: ['scatter_plot', 'line_plot', 'bar_chart', 'pie_chart', 'correlation_matrix'],
            multivariate: ['heatmap', '3d_scatter_plot', 'parallel_coordinate_plot', 'andrews_plot', 'radar_chart', 'chernoff_faces'],
            other: ['tree_map', 'sunburst_chart', 'force_directed_graph', 'sankey_diagram','covariance_matrix']
        };

        let previousButton = null;

        const chartTypeButtons = chartButtonsMap[hiddenInput.value];
if (chartTypeButtons) {
    chartTypeButtons.forEach(chartType => {
        const button = document.createElement('button');
        button.type = 'button';
        button.classList.add('chart-btn');
        button.value = chartType;
        button.textContent = chartType.replace('_', ' ');

        button.addEventListener('click', function () {
            // Clear the visualization result container
            visualizationResultsContainer.innerHTML = '';

            // Remove gray background from the previously clicked button
            if (previousButton) {
                previousButton.style.backgroundColor = '';
                previousButton.style.color = '';// Set it back to default
            }

            // Set the hidden input value to the clicked button's value
            hiddenInput.value = this.value;
            this.style.backgroundColor = '#808080';
            button.style.color = 'white';

            // Update the previousButton to the currently clicked button
            previousButton = this;
        });
        chartTypeButtonsContainer.appendChild(button);
            });
        }

        // Show the chart type buttons container
        chartTypeButtonsContainer.style.display = 'flex';
    }

});



  // Function to update the progress ring with a dynamic weightage (mean value)
function updateProgress(id, weightage) {
  const circle = document.getElementById(id);

  const radius = circle.clientWidth / 2;
  const circumference = 2 * Math.PI * radius;
  const progress = weightage / 100 * circumference;

  circle.innerHTML = `
      <svg class="progress-ring__circle" width="100" height="100">
          <circle class="progress-ring__circle-bg" stroke-width="10" fill="transparent" r="${radius}" cx="50" cy="50"></circle>
          <circle class="progress-ring__circle-progress" stroke-width="10" fill="transparent" r="${radius}" cx="50" cy="50" style="stroke-dasharray: ${progress} ${circumference}"></circle>
      </svg>
      <div class="progress-ring__text">${weightage.toFixed(2)}</div>
  `;
}

// Retrieve the mean value from the HTML template
const meanValue = parseFloat(document.getElementById('mean-value').textContent);

// Retrieve the median value from the HTML template
const medianValue = parseFloat(document.getElementById('median-value').textContent);

// Retrieve the mode value from the HTML template
const modeValue = parseFloat(document.getElementById('mode-value').textContent);

// Retrieve the Standard-deviation value from the HTML template
const standarddeviationValue = parseFloat(document.getElementById('stddev-value').textContent);

// Update the progress ring with the mean value
updateProgress('meanProgress', meanValue);
updateProgress('medianProgress', medianValue);
updateProgress('modeProgress', modeValue);
updateProgress('standarddeviationProgress',standarddeviationValue);
