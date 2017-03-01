package org.manuel.examples.RegressionAndClassificationExamples.weka;

import java.util.ArrayList;

import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.UnassignedClassException;
import weka.filters.Filter;

public class PolynomialRegression implements Classifier {
	weka.classifiers.functions.LinearRegression classifier;
	int degree; 
	/**
	 * Creates a classifier for performing polynomial regression of the degree specified 
	 * @param degree the degree of the polynomial model to build
	 */
	PolynomialRegression(int degree){
		classifier= new weka.classifiers.functions.LinearRegression();
		this.degree=degree;
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		if(data.classIndex()<0){
			throw new UnassignedClassException("Class index is negative (not set)!");
		}
		ArrayList<Attribute> attribs = new ArrayList<>(data.numAttributes() * degree);
		
		Instances newInstances = new Instances(data.relationName()+" ", attribs, data.numInstances());
		for (int i = 0; i < data.numInstances(); i++) {
			data.get(i);
		}
		/*FilteredClassifier f = new FilteredClassifier(); //se puede hacer usando otras clases
		Filter filter=;
		f.setClassifier(classifier);
		f.setFilter(filter);
		*/
	}

	@Override
	public double classifyInstance(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		return null;
	}
	
}
