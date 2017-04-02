package org.manuel.examples.RegressionAndClassificationExamples;

import java.awt.BorderLayout;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JFrame;
import javax.swing.JPanel;

import org.jfree.chart.ChartPanel;
import org.manuel.examples.RegressionAndClassificationExamples.weka.BayesianNetwork;
import org.manuel.examples.RegressionAndClassificationExamples.weka.NewLagMaker;
import org.manuel.examples.RegressionAndClassificationExamples.weka.PolynomialRegression;
import org.manuel.examples.RegressionAndClassificationExamples.weka.TimeSeries;
import org.manuel.examples.RegressionAndClassificationExamples.weka.Util;

import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.local.K2;
import weka.classifiers.timeseries.core.CustomPeriodicTest;
import weka.classifiers.timeseries.eval.ErrorModule;
import weka.classifiers.timeseries.eval.graph.JFreeChartDriver;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSink;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args ) throws Exception
    {
    	//testPolynomialRegression();
    	//testXml();
    	//testBayesNetWithWeather();
    	//testBayesNetWithWeather();
    	//testTimeSeries();
    	testTimeSeriesGraphic();
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
    static public void testXml(){
    	String nombre= "weather-10";
    	Instances data= Util.readXML(nombre+".xml");
    	System.out.println(data.toString());
    	try {
			DataSink.write(nombre+".arff", data);
		} catch (Exception e) {
			e.printStackTrace();
		}
    }
    public static void testBayesNetWithWeather(){
    	Instances data= Util.readXML("weather-10.xml");
    	Instances filteredData=BayesianNetwork.filterInstancesForBayesNet(data);
    	System.out.println("Filtered data:\n" +filteredData);
    	
    	BayesNet red= new BayesNet();
    	red.setEstimator(new SimpleEstimator());
    	red.setSearchAlgorithm(new K2());
    	try {
			red.buildClassifier(filteredData);
			String graph= red.graph();
			System.out.println(graph);
			BayesianNetwork.visualizeBayesNet(graph);
		} catch (Exception e) {
			e.printStackTrace();
		}
    }
    public static void testTimeSeries(){
    	try {
			Instances datos = Util.instancesFromFile("weather-10.arff");//Util.readXML("weather-10.xml");
			
			Instance last= datos.lastInstance();
			datos.delete(datos.numInstances()-1);
			
			TimeSeries timeseries = new TimeSeries();
			timeseries.setDebug(true);
			timeseries.setTimestamp("date");
			timeseries.setFieldsToForecast("windspeed");
			timeseries.setBaseLearner(new weka.classifiers.functions.LinearRegression());
			//timeseries.buildClassifier(datos);
			//double predicted = timeseries.classifyInstance(last);
			//System.out.print("#inst "+ datos.numInstances()+ " ");
			//System.out.println("actual: "+ last.value(datos.attribute("windspeed")) + " predicted: "+ predicted);
			int numInstPred=24;
			int size=datos.numInstances();
			Instances train = new Instances(datos, 0, size - numInstPred);
	        Instances test = new Instances(datos, size - numInstPred, numInstPred);
	        int attIndex= test.attribute("windspeed").index();
	        for (int i = 0; i < test.numInstances(); i++) {
	        	//test.get(i).setValue(attIndex, 0);
	        	//test.get(i).setMissing(attIndex);
			}
	        
			//Util.testTimeSeries(timeseries, datos, 24);
	        //Util.testTimeSeries(timeseries, train, test);
	        //timeseries.buildClassifier(train);
	        timeseries.classifyInstance(train, test);
			//Util.testTimeSeries(timeseries.forecaster, datos, null);//System.out
	        //timeseries.buildClassifier(train);
			Util.testTimeSeries(timeseries.forecaster, train, test, null);//System.out
		} catch (Exception e) {
			e.printStackTrace();
		}
    }
    public static void testTimeSeriesGraphic(){
    	try {
			Instances datos = Util.instancesFromFile("SOLAR_PRODUCTION_TRIMED.arff");//SOLAR_PRODUCTION.arff//Util.readXML("weather-10.xml");
			Instance last= datos.lastInstance();
			//datos.delete(datos.numInstances()-1);
			String targetName="kw" ;
			String xaxis="date";
			
			TimeSeries timeseries = new TimeSeries();
			timeseries.setDebug(true);
			timeseries.setTimestamp(xaxis);
			timeseries.setFieldsToForecast(targetName);
			timeseries.setBaseLearner(new weka.classifiers.functions.LinearRegression());
			
			
			//timeseries.getTSLagMaker().setAddAMIndicator(true);
			CustomPeriodicTest period= new CustomPeriodicTest(">=*:*:*:*:*:*:*:6:*:* <=*:*:*:*:*:*:*:18:*:*");
			period.getLowerTest().setOperator(">=");
			period.getLowerTest().setHourOfDay("6");
			period.getUpperTest().setOperator("<=");
			period.getUpperTest().setHourOfDay("18");
			String periodicTest="day="+ period.toString()+"";
			System.out.println(periodicTest);
			timeseries.forecaster.setTSLagMaker(new NewLagMaker());
			NewLagMaker lag=(NewLagMaker) timeseries.forecaster.getTSLagMaker();
			lag.setIncludePowersOfTime(false);
			lag.setIncludeTimeLagProducts(false);
			lag.setPrimaryPeriodicFieldName("hour"); //hour*************
			timeseries.forecaster.addCustomPeriodic(periodicTest);
			
			int numInstPred=24;
			int size=datos.numInstances();
			Instances train = new Instances(datos, 0, size - numInstPred);
	        Instances test = new Instances(datos, size - numInstPred, numInstPred);
	        int attIndex= test.attribute(targetName).index();
	        for (int i = 0; i < test.numInstances(); i++) {
	        	//test.get(i).setValue(attIndex, 0);
	        	//test.get(i).setMissing(attIndex);
			}
	        timeseries.setHorizon(24);
	        timeseries.setPrimeWindowSize(24);
	        //List<ErrorModule> preds= timeseries.classifyInstance(train, test);
	        List<ErrorModule> preds= timeseries.forecast(train, test);
			Util.testTimeSeries(timeseries.forecaster, train, test, null);//System.out
			
			JFreeChartDriver chart = new JFreeChartDriver();
			List<Integer> stepsToPlot= new ArrayList<>(2);
			stepsToPlot.add(1);
			//TODO also graph all the training data
			//JPanel graph= chart.getGraphPanelSteps(timeseries, preds, targetName, stepsToPlot, 0, train);
			JPanel graph= Util.graphSeries(preds.get(0), datos, targetName, xaxis);
			
			JFrame jf = new JFrame("Time series");
			jf.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
			jf.setSize(800, 600);
			jf.getContentPane().setLayout(new BorderLayout());
			jf.getContentPane().add(graph, BorderLayout.CENTER);
			jf.setVisible(true);
		} catch (Exception e) {
			e.printStackTrace();
		}
    }
}
