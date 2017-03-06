package org.manuel.examples.RegressionAndClassificationExamples;

import java.util.ArrayList;

import org.manuel.examples.RegressionAndClassificationExamples.weka.PolynomialRegression;
import org.manuel.examples.RegressionAndClassificationExamples.weka.Util;

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
    	testPolynomialRegression();
    }
    public static void testPolynomialRegression() throws Exception{
    	long time= System.currentTimeMillis();
    	PolynomialRegression classifier= new PolynomialRegression(2);
    	double datos[][]=new double[20][2];
    	for (int i = 0; i < datos.length; i++) {
			datos[i][0]= i;
			datos[i][1]= i*i+i+1 + 20 - Math.random()*40;
		}
    	Instances ins= Util.instancesFromArrays(datos);
    	classifier.buildClassifier(ins);
    	System.out.println("classifier build in "+ (System.currentTimeMillis() - time) + " ms");
    	
    	time= System.currentTimeMillis();
    	double y=0;
    	double x[]=new double[]{20, 400};
    	for (int i = 0; i < 1000; i++) {
    		y=classifier.classifyInstance(new DenseInstance(1.0, x));
		}
    	System.out.println(classifier.toString());
        System.out.println("x="+x[0]+" y= "+y);
        System.out.println("classified 1000 instances in "+ (System.currentTimeMillis() - time)+ " ms");
        Instances extrapolated=Util.extrapolate(ins, classifier, 5, 1);
        Util.compare(ins, extrapolated);
        //Util.plotData(ins);
    }
}
