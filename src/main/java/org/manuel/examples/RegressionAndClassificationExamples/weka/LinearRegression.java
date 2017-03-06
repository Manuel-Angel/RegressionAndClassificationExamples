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
		
		Instances data=Util.instancesFromFile("lineal.arff");
		//TODO check algorithmic complexity
		classifier.buildClassifier(data);

		System.out.println(classifier.toString());
		
		Instance inst= Util.instance(50,100, data);
		
		double r=classifier.classifyInstance(inst);
		System.out.println(inst.toString()+": "+ r);
		
		Util.plotData(data);
		
	}
	public static void exampleFromArray() throws Exception{
		//SimpleLinearRegression classifier= new SimpleLinearRegression();
		weka.classifiers.functions.LinearRegression classifier= new weka.classifiers.functions.LinearRegression();
		String info=classifier.globalInfo();
		System.out.println(info);
		
		String options[]= Utils.splitOptions("-output-debug-info");
		classifier.setOptions(options);
		
		Instances data= Util.instancesFromArrays(new double[][]{
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
		
		Instance inst= Util.instance(50.0,100.0, data);
		
		double r=classifier.classifyInstance(inst);
		System.out.println(inst.toString()+": "+ r);
	}
	
}
