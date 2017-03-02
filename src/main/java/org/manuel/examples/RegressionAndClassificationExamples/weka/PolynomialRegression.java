package org.manuel.examples.RegressionAndClassificationExamples.weka;

import java.util.ArrayList;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.net.search.fixed.FromFile;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.UnassignedClassException;
import weka.filters.Filter;

public class PolynomialRegression implements Classifier {
	weka.classifiers.functions.LinearRegression classifier;
	Instances instances;
	int degree; 
	/**
	 * Creates a classifier for performing polynomial regression of the degree specified 
	 * @param degree the degree of the polynomial model to build
	 */
	public PolynomialRegression(int degree){
		classifier= new weka.classifiers.functions.LinearRegression();
		this.degree=degree;
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		if(data.classIndex()<0){
			throw new UnassignedClassException("Class index is negative (not set)!");
		}
		ArrayList<Attribute> attribs = new ArrayList<>((data.numAttributes() -1) * degree);
		int att;
		for (att = 0; att < data.numAttributes(); att++) {
			attribs.add(data.attribute(att));
		}
		
		for (int j = 0; j < data.numAttributes(); j++) {
			if(data.classIndex()==j)
				continue;
			
			for (int d = 2; d <= degree; d++) {
				StringBuilder sb= new StringBuilder(data.attribute(j).name());
				sb.append("_").append(d);
				
				attribs.add(new Attribute(sb.toString()));
			}
		}
		
		Instances newInstances = new Instances(data.relationName()+"polynomial", attribs, data.numInstances());
		for (int in = 0; in < data.numInstances(); in++) {
			Instance inst= data.get(in);
			/*double values[]=new double[newInstances.numAttributes()];
			
			for (att = 0; att < inst.numAttributes(); att++) {
				values[att]=inst.value(att);
			}
			
			for(int j=0;j<inst.numAttributes();j++){
				if(inst.classIndex() == j){
					continue;
				}
				double aux= values[j];
				for (int d= 2; d <= degree; d++) {
					aux*=values[j];
					values[att++]= aux;
				}
			}
			
			Instance newInst = new DenseInstance(1.0, values);*/
			Instance newInst = changeInstance(inst, newInstances.numAttributes());
			newInstances.add(newInst);
		}
		instances= newInstances;
		instances.setClassIndex(data.classIndex());
		classifier.buildClassifier(newInstances);
		
		/*FilteredClassifier f = new FilteredClassifier(); //se puede hacer usando otras clases
		Filter filter=;
		f.setClassifier(classifier);
		f.setFilter(filter);
		*/
	}
	protected Instance changeInstance(Instance inst, int numAtt){
		double values[]=new double[numAtt];
		int att;
		for (att = 0; att < inst.numAttributes(); att++) {
			values[att]=inst.value(att);
		}
		for(int j=0;j<inst.numAttributes();j++){
			if(inst.classIndex() == j){
				continue;
			}
			double aux= values[j];
			for (int d= 2; d <= degree; d++) {
				aux*=values[j];
				values[att++]= aux;
			}
		}
		Instance newInst = new DenseInstance(1.0, values);
		return newInst;
	}
	@Override
	public double classifyInstance(Instance instance) throws Exception {
		instance.setDataset(instances);
		Instance newInst= changeInstance(instance,instances.numAttributes());
		double coeff[]= classifier.coefficients();
		
		
		return classifier.classifyInstance(newInst);
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		return classifier.distributionForInstance(instance);
	}

	@Override
	public Capabilities getCapabilities() {
		//TODO Auto-generated method stub
		return classifier.getCapabilities();
	}
	
}
