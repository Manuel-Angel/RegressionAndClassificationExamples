package org.inaoe.examples.RegressionAndClassificationExamples.weka;

import java.io.PrintStream;

import weka.classifiers.functions.*;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Utils;

public class LinearRegression {
	public static void main(String[] args) throws Exception {
		System.setErr(new PrintStream("errors")); //to not get warnings in standar output
		
		SimpleLinearRegression classifier= new SimpleLinearRegression();
		String info=classifier.globalInfo();
		System.out.println(info);
		
		String options[]= Utils.splitOptions("-output-debug-info");
		classifier.setOptions(options);
		
		Instances data=instanceFromFile();
		
		classifier.buildClassifier(data);
		
		double pendiente= classifier.getSlope();
		double b = classifier.getIntercept();
		System.out.println(pendiente+ " "+ b);
		
		Instance inst= instance(50,100, data);
		
		double r=classifier.classifyInstance(inst);
		System.out.println(inst.toString()+": "+ r);
	}
	public static Instances instanceFromFile() throws Exception{
		Instances data= DataSource.read("lineal.arff");
		data.setClassIndex(data.numAttributes()-1);
		return data;
	}
	public static Instance instance(double a, double b, Instances data){
		Attribute atributo_a= data.attribute(0), atributo_b= data.attribute(1);
		
		Instance inst= new DenseInstance(2);
		inst.setDataset(data);
		inst.setValue(atributo_a, a);
		inst.setValue(atributo_b, b);
		return inst;
	}
}
