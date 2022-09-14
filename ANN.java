package main;

import java.util.ArrayList;
import java.util.Arrays;

public class ANN {
	//new int[2][4]; Two rows and four columns
	//rows = arr.length; //columns = arr[0].length;
	private ArrayList<double[][]> sums;
	private ArrayList<double[][]> weights;
	private ArrayList<double[][]> biases;
	private ArrayList<double[][]> activationGradients;
	private ArrayList<double[][]> sumsGradients;
	private ArrayList<double[][]> weightsGradients;
	private ArrayList<double[][]> biasesGradients;
	private int inputSize;
	private enum Function {
		Sigmoid,
		ArcTan,
		ReLu,		
	};
	private Function func;
	
	public ANN(Function func, int... settings) {
		this.func = func;
		this.inputSize = settings[0]; // Dimensions of input vector
		
		this.sums = new ArrayList<double[][]>(settings.length); // # of weight matrices = # of layers - 1
		for(int i = 0; i < settings.length-1; i++) {
			this.sums.add(new double[settings[i+1]][1]); 
		}
		
		this.weights = new ArrayList<double[][]>(settings.length-1); // # of weight matrices = # of layers - 1
		for(int i = 0; i < settings.length-1; i++) {
			this.weights.add(this.initArray(settings[i+1], settings[i]));
		}
		
		this.biases = new ArrayList<double[][]>(settings.length-1); // # of bias matrices = # of layers - 1
		for(int i = 0; i < settings.length-1; i++) {
			this.biases.add(this.initArray(settings[i+1], 1));
		}

		this.printState();
	}
	
	/**
	* Runs math calculations using given input.
	* 
	* @param input the input vector into the matrix
	* @return the vector of the NN prediction
	*/
	private double[][] forwardPropagation(double[][] input) {
        if(input.length != this.inputSize || input[0].length != 1) {
            throw new IllegalArgumentException("Input Error!"); //colA must equal rowB
        }
		sums.set(0, this.add(this.multiply(this.weights.get(0), input), this.biases.get(0)));
		for(int i = 1; i < this.sums.size(); i++) {
			sums.set(i, this.add(this.multiply(this.weights.get(i), this.activate(this.sums.get(i-1))), this.biases.get(i)));
		}
		return this.activate(this.sums.get(this.sums.size()-1));
	}
	
	/**
	* Runs back propagation algorithm to adjust weights and biases
	* 
	* @param input the input vector into the matrix
	*/
	private double[][] backPropogation(double[][] input) {
		return null;
	}
	
	/**
	* Computes the error in the NN prediction
	* 
	* @param input the input vector into the NN
	* @param expectedOutput the vector of the expected output of the NN
	* @return the cost of the NN
	*/
	private double cost(double[][] input, double[][] expectedOutput) {
        if(input.length != this.inputSize || input[0].length != 1 || expectedOutput[0].length != 1) {
            throw new IllegalArgumentException("Input Error!"); //colA must equal rowB    //rows = arr.length; //columns = arr[0].length;
        }
        double cost = 0;
        double[][] output = this.forwardPropagation(input);
        
        if(output.length != expectedOutput.length) {
            throw new IllegalArgumentException("Input Error!"); //colA must equal rowB    //rows = arr.length; //columns = arr[0].length;
        }
        
        for(int i = 0; i < output.length; i++) {
        	cost += ((output[i][0] - expectedOutput[i][0]) * (output[i][0] - expectedOutput[i][0]));
        }
		return cost;
	}

	public static void main(String[] args) {
		System.out.println("Running");
		ANN ann = new ANN(Function.Sigmoid, new int[] {2,3,2});
		System.out.println("Prediction: " + Arrays.deepToString(ann.forwardPropagation(new double[][] {{1},{-1}})));
		System.out.println("Cost: " + ann.cost(new double[][] {{2},{3}}, new double[][] {{1},{0}}));
		
//		System.out.println("Testing Matrix Multiply");
//		double[][] A = new double[][] {
//			{1,2,3},
//			{4,5,6}
//		};
//		
//		double[][] B = new double[][] {
//			{10, 11},
//			{20, 21},
//			{30, 31}
//		};
//		System.out.println(Arrays.deepToString(ann.multiply(A, B)));
		
//		System.out.println("Testing Matrix Add");
//		double[][] A = new double[][] {
//			{1,2,3},
//			{4,5,6}
//		};
//		double[][] B = new double[][] {
//			{-1.5},
//			{-1.5},
//			{-1.5}
//		};
//		System.out.println(Arrays.deepToString(ann.add(A, B)));
		
//		System.out.println("Testing Matrix Activation");
//		double[][] A = new double[][] {
//			{-1,1,0},
//			{0,0,0}
//		};
//		System.out.println(Arrays.deepToString(ann.activation(A)));
	}
	
	/**
	* Passes the parameter through the programs chosen activation function
	* @param x the value being passed through the function
	* @return the value after being passed through the function
	*/
	private double f(double x) {
		if(this.func == Function.Sigmoid) {
			return (1.0 / (1.0 + Math.exp(-x)));
		}
		if(this.func == Function.ArcTan) {
			return Math.atan(x);
		}
		if(this.func == Function.ReLu) {
			return (x > 0) ? x : 0;
		}
		throw new IllegalArgumentException("f Error!");
	}
	
	/**
	* Passes the parameter through the derivative of the programs chosen activation function
	* @param x the value being passed through the derivative of the activation function
	* @return the value after being passed through the derivative of the activation function
	*/
	private double fPrime(double x) {
		if(this.func == Function.Sigmoid) {
			return (this.f(x) * (1.0 - this.f(x)));
		}
		if(this.func == Function.ArcTan) {
			return (1.0 / ((x*x) + 1.0));
		}
		if(this.func == Function.ReLu) {
			return (x > 0) ? 1 : 0;
		}
		throw new IllegalArgumentException("fPrime Error!");
	}
	
	/**
	* Statistical normal distribution function
	* @param x the value being passed through the function
	* @return the value after being passed through the function
	*/
	private double normalDistribution(double x) {
		return (Math.exp(-(x*x)/2.0)/Math.sqrt(2*Math.PI));
	}
	
	/**
	* Creates a row by cols array set to a random value from the normal distribution
	* @param rows the number of rows in the matrix
	* @param cols the number of columns in the matrix
	* @return the newly initialized array
	*/
	private double[][] initArray(int rows, int cols) {
		double[][] C = new double[rows][cols];
    	for(int i = 0; i < rows; i++) {
    		for(int j = 0; j < cols; j++) {
    			C[i][j] = this.normalDistribution(-4.0 + (Math.random() * 8.0));
    		}
    	}
		return C;
	}
	
	/**
	* Computes the matrix multiplication of A with B
	* 
	* @param A the first matrix that is multiplying the second
	* @param B the second matrix
	* @return the matrix multiplication of A with B
	*/
	private double[][] multiply(double[][] A, double[][] B) {
        if(A[0].length != B.length) {
            throw new IllegalArgumentException("Multiplication Error!"); //colA must equal rowB    //rows = arr.length; //columns = arr[0].length;
        }
        double[][] C = new double[A.length][B[0].length];
        for(int i = 0; i < A.length; i++) {
        	for(int j = 0; j < B[0].length; j++) {
        		for(int k = 0; k < A[0].length; k++) {
        			C[i][j] += (A[i][k] * B[k][j]);
        		}
        	}
        }
        return C;
	}
	
	/**
	* Computes the matrix addition of A and B
	* 
	* @param A the first matrix
	* @param B the Vector being added
	* @return the matrix addition of A and B
	*/
	private double[][] add(double[][] A, double[][] B) {
        if(A.length != B.length || B[0].length != 1) {
            throw new IllegalArgumentException("Addition Error!"); //colA must equal rowB    //rows = arr.length; //columns = arr[0].length;
        }
        double[][] C = new double[A.length][A[0].length];
        for(int i = 0; i < C.length; i++) {
            for(int j = 0; j < C[0].length; j++) {
            	C[i][j] = (A[i][j] + B[i][0]);
            }
        }
        return C;
	}
	
	/**
	* Applies the activation function to the matrix
	* 
	* @param A the matrix
	* @return the matrix that has been activated
	*/
	private double[][] activate(double[][] A) {
        double[][] C = new double[A.length][A[0].length];
        for(int i = 0; i < C.length; i++) {
            for(int j = 0; j < C[0].length; j++) {
            	C[i][j] = this.f(A[i][j]);
            }
        }
        return C;
	}
	
	/**
	* Prints the states of all the matrices in the NN
	*/
	private void printState() {
		System.out.println("\nInputs:");
		System.out.println(this.inputSize);
		
		System.out.println("\nsums:");
		System.out.println(this.sums.size());
		for(double[][] arr: this.sums) {
			System.out.println(Arrays.deepToString(arr));
		}
		
		System.out.println("\nWeights:");
		System.out.println(this.weights.size());
		for(double[][] arr: this.weights) {
			System.out.println(Arrays.deepToString(arr));
		}
		
		System.out.println("\nBiases:");
		System.out.println(this.biases.size());
		for(double[][] arr: this.biases) {
			System.out.println(Arrays.deepToString(arr));
		}
	}

}