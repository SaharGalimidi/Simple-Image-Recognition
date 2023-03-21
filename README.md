<!DOCTYPE html>
<html>
<head>
</head>
<body>
	<h1>Simple Image Recognition</h1>
	<p>🌟 This is a parallelized implementation of a simplified image recognition algorithm using MPI, OpenMP, and CUDA. 🔍🖼️</p>
	<h2>Project Description</h2>
	<p>📝 The project deals with sets of pictures and objects of different sizes. Each member of the matrix represents a color, and the range of possible colors is [1, 100]. The goal of the project is to find an object within a picture using a matching algorithm that calculates the total difference between overlapping members of the picture and the object.</p>
	<h2>Technologies Used</h2>
	<ul>
		<li>MPI - Message Passing Interface, used to dynamically assign a task (picture) to each slave, which in turn processes the objects on the given picture and sends the log results back to the master.</li>
		<li>OpenMP - Open Multi-Processing, used to divide the objects on each picture into OpenMP threads. Each thread checks a specific object on the picture.</li>
		<li>CUDA - Compute Unified Device Architecture, used to calculate all possible locations of the given object on the picture in parallel.</li>
	</ul>
	<h2>Requirements</h2>
	<p>📋 To run this project, you will need:</p>
	<ul>
		<li>C++ compiler with OpenMP support 🖥️💻</li>
		<li>MPI library 📚</li>
		<li>CUDA Toolkit 🛠️</li>
	</ul>
<h2>How to Run</h2>
  <ol>
      <li>Clone the repository: <code>git clone https://github.com/SaharGalimidi/Simple-Image-Recognition.git</code></li>
      <li>Compile the code: <code>make</code></li>
      <li>Run the code with at least 2 MPI processes: <code>make run</code></li>
  </ol>
	<h2>Output Format</h2>
	<p>📄 The output file will contain the results of the recognition algorithm for each picture. For each picture, the log will indicate whether at least three objects were found with an appropriate matching value. If three objects were found, the log will also include the starting position of each object in the picture.</p>
  <h2>Performance</h2>
<p>The parallelized implementation using MPI, OpenMP, and CUDA significantly improves the performance of the image recognition algorithm. The original sequential implementation took approximately 60 seconds to complete, while the parallelized implementation reduces this time to approximately 0.7 seconds.</p>
</body>
</html>
