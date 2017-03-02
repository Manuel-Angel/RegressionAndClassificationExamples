package org.manuel.examples.RegressionAndClassificationExamples;

import java.util.ArrayList;

import org.manuel.examples.RegressionAndClassificationExamples.weka.PolynomialRegression;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args ) throws Exception
    {
    	PolynomialRegression classifier= new PolynomialRegression(2);
    	double datos[][]=new double[15][2];
    	for (int i = 0; i < datos.length; i++) {
			datos[i][0]= i;
			datos[i][1]= i*i;
		}
    	Instances ins= instancesFromArrays(datos);
    	classifier.buildClassifier(ins);
    	
    	double x[]=new double[]{20, 400};
    	double y=classifier.classifyInstance(new DenseInstance(1.0, x));
    	
        System.out.println("y= "+y);
    }
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
    public static Instance instance(double values[], Instances data){
		Instance inst = new DenseInstance(1.0, values);
		inst.setDataset(data);//this is optional, just for using methods about the attributes
		return inst;
	}
}
