package org.manuel.examples.RegressionAndClassificationExamples.weka;

import java.awt.BorderLayout;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Random;

import javax.swing.JFrame;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.functions.*;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;
import weka.core.Utils;

public class LinearRegression {
	public static void main(String[] args) throws Exception {
		//System.setErr(new PrintStream("errors")); //to not get warnings in standar output
		//TODO hacerlo con el modelo factory
		System.out.println("Example from file:");
		exampleFromFile();
		System.out.println("Example from array:");
		exampleFromArray();
	}
	
	public static void exampleFromFile() throws Exception{
		//SimpleLinearRegression classifier= new SimpleLinearRegression();
		weka.classifiers.functions.LinearRegression classifier= new weka.classifiers.functions.LinearRegression();
		String info=classifier.globalInfo();
		System.out.println(info);
		
		String options[]= Utils.splitOptions("-output-debug-info");
		classifier.setOptions(options);
		
		Instances data=instancesFromFile("lineal.arff");
		//TODO check algorithmic complexity
		classifier.buildClassifier(data);

		System.out.println(classifier.toString());
		
		Instance inst= instance(50,100, data);
		
		double r=classifier.classifyInstance(inst);
		System.out.println(inst.toString()+": "+ r);
		
		Evaluation eval =new Evaluation(data);
		eval.crossValidateModel(classifier, data, 10, new Random(1));
		
		
		//ThresholdCurve tc = new ThresholdCurve();
		Instances curve = data;//tc.getCurve(eval.predictions(), 1);
		
		PlotData2D plotdata = new PlotData2D(curve);
		plotdata.setPlotName(curve.relationName());
		plotdata.addInstanceNumberAttribute();
		
		ThresholdVisualizePanel tvp = new ThresholdVisualizePanel();
		tvp.setROCString("(Area under ROC = " +
				Utils.doubleToString(ThresholdCurve.getROCArea(curve),4)+")");
		tvp.setName(curve.relationName());
		tvp.addPlot(plotdata);
		
		JFrame jf = new JFrame("WEKA ROC: " + tvp.getName());
		jf.setSize(500,400);
		jf.getContentPane().setLayout(new BorderLayout());
		jf.getContentPane().add(tvp, BorderLayout.CENTER);
		jf.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		jf.setVisible(true);
	}
	public static void exampleFromArray() throws Exception{
		//SimpleLinearRegression classifier= new SimpleLinearRegression();
		weka.classifiers.functions.LinearRegression classifier= new weka.classifiers.functions.LinearRegression();
		String info=classifier.globalInfo();
		System.out.println(info);
		
		String options[]= Utils.splitOptions("-output-debug-info");
		classifier.setOptions(options);
		
		Instances data= instancesFromArrays(new double[][]{
			{1, 2.1},
			{2, 4.01},
			{3, 5.9},
			{4, 7.99},
			{5, 10},
			{6, 11.9},
			{7, 13.99},
			{8, 16.2},
			{9, 18.1},
			{10, 20},
			{11, 22.019},
			{12, 23.9},
			{13, 26},
			{14, 28},
			{15, 29.99}
			});
		
		classifier.buildClassifier(data);

		System.out.println(classifier.toString());
		
		Instance inst= instance(50,100, data);
		
		double r=classifier.classifyInstance(inst);
		System.out.println(inst.toString()+": "+ r);
	}
	/**
	 * Reads the data from a file and returns an
	 * object of class Instances with the data from the file.
	 * Sets the last variable as the class variable, that is the
	 * variable to predict.
	 * @param file The file where the data is extracted
	 * @return an object of class Instances with the data
	 * @throws Exception if file is not found
	 */
	public static Instances instancesFromFile(String file) throws Exception{
		Instances data= DataSource.read(file);
		data.setClassIndex(data.numAttributes()-1);
		return data;
	}
	/**
	 * Creates a Instances object from the values where the first index is the index of instance and the 
	 * second index is the attribute index. It names the attributes xi (x0,x1,x2...)
	 * @param values the values of the instances
	 * @return Instances object with the values from the array values
	 */
	
	public static Instances instancesFromArrays(double [][]values) {
		ArrayList<Attribute> attribs = new ArrayList<>(values[0].length);
		for (int i = 0; i < values[0].length-1; i++) {
			attribs.add(new Attribute("x" + i));
		}
		attribs.add(new Attribute("y"));
		
		Instances inst= new Instances("lineal", attribs, values.length);
		inst.setClassIndex(values[0].length-1);
		
		for (int i = 0; i < values.length; i++) {
			inst.add(instance(values[i],inst));
		}
		return inst;
	}
	/**
	 * Creates a Instances object from the values where the first index is the index of instance and the 
	 * second index is the attribute index, and names the attributes with the strings in the labels array. 
	 * @param values
	 * @param labels
	 * @return
	 */
	public static Instances instancesFromArrays(double [][]values, String labels[]) {
		ArrayList<Attribute> attribs = new ArrayList<>(values[0].length);
		for (int i = 0; i < values[0].length; i++) {
			attribs.add(new Attribute(labels[i]));
		}
		Instances inst= new Instances("lineal", attribs, values.length);
		inst.setClassIndex(values[0].length-1);
		for (int i = 0; i < values.length; i++) {
			inst.add(instance(values[i],inst));
		}
		return inst;
	}
	/**
	 * Creates and returns an object of class Instance with the given parameters, with the info about the dataset.
	 * This method is less efficient than instance(double values[], Instance data)
	 * @param a the independent variable
	 * @param b the dependent variable or class variable 
	 * @param data the data set
	 * @return
	 */
	public static Instance instance(double a, double b, Instances data){
		//Attribute atributo_a= data.attribute(0), atributo_b= data.attribute(1);
		if(data.classIndex()==0){
			double aux=a;
			a=b;
			b=aux;
		}
		Instance inst= new DenseInstance(2);
		inst.setDataset(data); //this is optional, just for using methods about the attributes
		//inst.setValue(atributo_a, a); //The attribute has to belong to a dataset
		//inst.setValue(atributo_b, b);
		inst.setValue(0, a);
		inst.setValue(1, b);
		return inst;
	}
	/**
	 * Creates and returns an object of class Instance with the given values, with the info about the dataset.
	 * The values arrays contains the independent(s) and dependent variable, and must be compatible with the dataset 
	 * (same number of variables and in the same order).
	 * @param values 
	 * @param data
	 * @return
	 */
	public static Instance instance(double values[], Instances data){
		Instance inst = new DenseInstance(1.0, values);
		inst.setDataset(data);//this is optional, just for using methods about the attributes
		return inst;
	}
}
