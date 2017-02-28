package org.manuel.examples.RegressionAndClassificationExamples.weka;

import java.io.PrintStream;
import java.util.ArrayList;

import weka.classifiers.functions.*;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Utils;

public class LinearRegression {
	public static void main(String[] args) throws Exception {
		//System.setErr(new PrintStream("errors")); //to not get warnings in standar output
		exampleFromFile();
	}
	
	public static void exampleFromFile() throws Exception{
		SimpleLinearRegression classifier= new SimpleLinearRegression();
		String info=classifier.globalInfo();
		System.out.println(info);
		
		String options[]= Utils.splitOptions("-output-debug-info");
		classifier.setOptions(options);
		
		Instances data=instancesFromFile("lineal.arff");
		//data.add(instance(40,80,data));
		
		classifier.buildClassifier(data);
		
		double pendiente= classifier.getSlope();
		double b = classifier.getIntercept();
		System.out.println(pendiente+ " "+ b);
		
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
	public static Instances instancesFromArrays(double [][]values) {
		Attribute a=new Attribute("a");
		Attribute b=new Attribute("b");
		ArrayList<Attribute> attribs = new ArrayList<>(2);
		attribs.add(a);
		attribs.add(b);
		
		Instances inst= new Instances("lineal", attribs, values.length);
		inst.setClass(b);
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
	 * Creates and returns an object of class Instance with the given parameters, with the info about the dataset.
	 * The values arrays contains the independent and dependent variable respectively
	 * @param values
	 * @param data
	 * @return
	 */
	public static Instance instance(double values[], Instances data){
		if(data.classIndex()==0){
			double a=values[0];
			values[0]=values[1];
			values[1]=a;
		}
		Instance inst = new DenseInstance(1.0, values);
		inst.setDataset(data);//this is optional, just for using methods about the attributes
		return inst;
	}
}
