package org.manuel.examples.RegressionAndClassificationExamples.weka;

import java.awt.BorderLayout;
import java.awt.Color;
import java.io.IOException;
import java.io.PrintStream;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

import javax.swing.JFrame;
import javax.swing.JPanel;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.local.K2;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.NumericPrediction;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.timeseries.TSForecaster;
import weka.classifiers.timeseries.core.OverlayForecaster;
import weka.classifiers.timeseries.eval.ErrorModule;
import weka.classifiers.timeseries.eval.TSEvaluation;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.graphvisualizer.BIFFormatException;
import weka.gui.graphvisualizer.GraphVisualizer;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

import javax.xml.parsers.*;
import javax.xml.transform.*;
import javax.xml.transform.dom.*;
import javax.xml.transform.stream.*;

import org.xml.sax.*;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.DateAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYErrorRenderer;
import org.jfree.chart.title.TextTitle;
import org.jfree.data.xy.XYIntervalSeries;
import org.jfree.data.xy.XYIntervalSeriesCollection;
import org.w3c.dom.*;

public class Util {
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
		
		Instances inst= new Instances("dataset", attribs, values.length);
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
		Instances inst= new Instances("dataset", attribs, values.length);
		inst.setClassIndex(values[0].length-1);
		for (int i = 0; i < values.length; i++) {
			inst.add(instance(values[i],inst));
		}
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
	 * Plots a single data set from an object of class Instances
	 * @param data the data set to be plotted
	 * @throws Exception
	 */
	public static void plotData(Instances data) throws Exception{
		//Evaluation eval =new Evaluation(data);
		//eval.crossValidateModel(classifier, data, 10, new Random(1));
		//ThresholdCurve tc = new ThresholdCurve();
		Instances curve = data;//tc.getCurve(eval.predictions(), 1);
		
		PlotData2D plotdata = new PlotData2D(curve);
		plotdata.setPlotName(curve.relationName());
		//plotdata.addInstanceNumberAttribute();
		plotdata.setCustomColour(new Color(0,100,200));
		
		plotData(plotdata);
	}
	public static void plotData(PlotData2D plotdata) throws Exception{		
		plotData(new PlotData2D[]{plotdata});
	}
	public static void plotData(PlotData2D []plotdata) throws Exception{
		ThresholdVisualizePanel tvp = new ThresholdVisualizePanel();
		tvp.setName(plotdata[0].getPlotName());
		//tvp.setROCString("(Area under ROC = " +
			//Utils.doubleToString(ThresholdCurve.getROCArea(curve),4)+")");
		for (int i = 0; i < plotdata.length; i++) {
			tvp.addPlot(plotdata[i]);			
		}
		JFrame jf = new JFrame("Plot for: " + tvp.getName());
		jf.setSize(500,400);
		jf.getContentPane().setLayout(new BorderLayout());
		jf.getContentPane().add(tvp, BorderLayout.CENTER);
		jf.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		jf.setVisible(true);
	}
	public static Instances extrapolate(Instances data, Classifier classifier, int numberOfTests, double step) throws Exception{
		//Evaluation eval =new Evaluation(data);
		//eval.crossValidateModel(classifier, data, 10, new Random(1));
		//ThresholdCurve tc = new ThresholdCurve();
		//Instances curve = data;
		//tc.getCurve(eval.predictions(), 1);
		
		Instances newData= new Instances(data, data.numInstances() + numberOfTests);
		newData.setRelationName(data.relationName() + "_extrapolated");
		for (int i = 0; i < data.numInstances(); i++) {
			Instance inst= (Instance)data.instance(i).copy();
			inst.setClassValue(classifier.classifyInstance(inst));
			newData.add(inst);
		}
		Instance last= data.get(data.numInstances()-1);
		double lasts[] = new double[data.numAttributes()];
		for (int i = 0; i < data.numAttributes(); i++) {
			lasts[i]= last.value(i);
		}
		for (double i = 0; i < numberOfTests; i+=step) {
			double values[]= new double[data.numAttributes()];
			for (int j = 0; j < values.length; j++) {
				lasts[j]+=step;
				values[j]= lasts[j];
			}
			Instance inst= instance(values, newData);
			inst.setClassValue(classifier.classifyInstance(inst));
			newData.add(inst);
		}
		return newData;
	}
	public static void compare(Instances datasets[]){
		PlotData2D plots[]= new PlotData2D[datasets.length];
		for (int i = 0; i < datasets.length; i++) {
			plots[i]= new PlotData2D(datasets[i]);
			plots[i].setPlotName(datasets[i].relationName());
		}
		
	}
	public static void compare(Instances original, Instances test) throws Exception{
		PlotData2D plots[]= new PlotData2D[2];
		plots[0]= new PlotData2D(original);
		plots[0].setPlotName(original.relationName());
		plots[0].setCustomColour(new Color(0,100,200));
		
		plots[1]= new PlotData2D(test);
		plots[1].setPlotName(test.relationName());
		plots[1].setCustomColour(new Color(200,0,10));
		boolean cp[]= new boolean[test.numInstances()];
		Arrays.fill(cp, true);
		plots[1].setConnectPoints(cp);
		plotData(plots);
	}
	/**
	 * Creates an Instances object with the header for storing the weather data.
	 * @param size the initial capacity of the Instances object
	 * @return an Instances object with the format for storing weather data. 
	 */
	public static Instances createWeatherInstances(int size){
		ArrayList<Attribute> atts = new ArrayList<>();
		atts.add(new Attribute("cloudcover"));
		atts.add(new Attribute("date", "yyyy-MM-dd HH:mm"));
		atts.add(new Attribute("temp"));
		atts.add(new Attribute("winddir"));
		atts.add(new Attribute("windspeed"));
		Instances in= new Instances("weatherReports", atts, size);
		
		in.setClassIndex(4); //the class attribute is windspeed
		return in;
	}
	public static Instances readXML(String xml){
		Document dom;
        // Make an  instance of the DocumentBuilderFactory
        DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
        DocumentBuilder db;
        Instances inst= createWeatherInstances(100*24);
		try {
			db = dbf.newDocumentBuilder();
			// parse using the builder to get the DOM mapping of the    
	        // XML file
			dom = db.parse(xml);
	        Element doc = dom.getDocumentElement();
	        Node reportes= doc.getElementsByTagName("weatherReports").item(0);
	        
	        String tag= reportes.getNodeName();
	        
	        NodeList nodos= reportes.getChildNodes();
	        //System.out.println("Elemento: " + tag + "  valor: " + reportes.getNodeValue());
	        for (int i = 0; i < nodos.getLength(); i++) {
				Node nodo= nodos.item(i);
				//System.out.println(nodo.getNodeName()+ " - "+ nodo.getNodeValue() +" - " + nodo.getNodeType());
				NamedNodeMap atributos= nodo.getAttributes();
				if(nodo.getNodeType() !=Node.ELEMENT_NODE)continue;
				//if(atributos==null) continue;
				double values[]= new double[5];
				String date="";
				try{
					for (int j = 0; j < atributos.getLength(); j++) {
						Node atributo=atributos.item(j);
						//System.out.println(atributo.getNodeName() +"= \"" + atributo.getNodeValue() + "\" ");
						switch(atributo.getNodeName()){
							case "cloudcover":  values[0]= Double.parseDouble(atributo.getNodeValue()); break;
							case "date":
									date=atributo.getNodeValue();
									try{
										values[1]= inst.attribute(1).parseDate(date); // this has m_DateFormat.setLenient(false); so it might throw format exception, for example with date 2009-04-05 02:00, which is valid 
									}catch(ParseException e){
										SimpleDateFormat format= new SimpleDateFormat("yyyy-MM-dd HH:mm");
										values[1]=format.parse(date).getTime();
									}
								break;
							case "temp":		values[2]= Double.parseDouble(atributo.getNodeValue()); break;
							case "winddir": 	values[3]= Double.parseDouble(atributo.getNodeValue()); break;
							case "windspeed": 	values[4]= Double.parseDouble(atributo.getNodeValue()); break;
							default: throw new java.text.ParseException("Attribute does not exist", 1);
						}
						
					}
				}catch (ParseException e) {
					System.out.println("error parseando " +date+" elemento "+i+" sera ingorado\n" + e);
					e.printStackTrace();
					continue;
				}
				Instance act=instance(values,inst);
				inst.add(act);
			}
		} catch (ParserConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (SAXException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (DOMException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		inst.compactify();
		return inst;
	}
	public static void testTimeSeries(TSForecaster forecaster,Instances data, PrintStream progress){
		try {
			TSEvaluation evaluation= new TSEvaluation(data, 24);
			testTimeSeries(forecaster, evaluation, progress);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	public static void testTimeSeries(TSForecaster forecaster, Instances training, Instances test, PrintStream progress){
		try {
			TSEvaluation evaluation= new TSEvaluation(training, test);
			testTimeSeries(forecaster, evaluation, progress);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	protected static void testTimeSeries(TSForecaster forecaster, TSEvaluation evaluation, PrintStream progress) throws Exception{
		PrintStream prog[];
		if(progress==null){
			prog= new PrintStream[0];
		} else prog= new PrintStream[]{progress};
		evaluation.setHorizon(24);
		evaluation.setPrimeWindowSize(24);
		//evaluation.setPrimeWindowSize(0);
		evaluation.evaluateForecaster(forecaster, true,prog);
		//evaluation.setEvaluateOnTrainingData(false);
		String field=forecaster.getFieldsToForecast();
		System.out.println("evaluate with testing/training data " + evaluation.getEvaluateOnTestData()+
				" " + evaluation.getEvaluateOnTrainingData());
		System.out.println("prime window size: " + evaluation.getPrimeWindowSize());
		System.out.println("priming with test data " +evaluation.getPrimeForTestDataWithTestData());
		System.out.println("is using overlay data" + (evaluation instanceof OverlayForecaster));

		System.out.println("target: "+ field);
		System.out.println(evaluation.printPredictionsForTestData("titulo", field, 1));
		System.out.println(evaluation.toSummaryString());
	}
	public static void testTimeSeries(TimeSeries forecaster, Instances trainingData, int numInstPred){
		int size=trainingData.numInstances();
		Instances train = new Instances(trainingData, 0, size - numInstPred);
        Instances test = new Instances(trainingData, size - numInstPred, numInstPred);
        testTimeSeries(forecaster, train, test);
	}
	public static void testTimeSeries(TimeSeries forecaster, Instances train, Instances test){
		String target=forecaster.getFieldsToForecast();
        Attribute att=train.attribute(target);
        double sum = 0, error;
        try {
			forecaster.buildClassifier(train);
			System.out.printf("%12s %12s %12s %12s\n", "inst#", "actual","predicted","error");
			double pred, act;
			forecaster.setDebug(false);
			for (int i = 0; i < test.numInstances(); i++) {
				Instance testInt = test.get(i);
				pred= forecaster.classifyInstance(testInt);
				act=testInt.value(att);
				error=pred -act;
				sum+=Math.abs(error);
				System.out.printf("%12d %12f %12f %12f\n", i+1,act ,pred,error );
			}
			System.out.println("Mean absolute error: "+ (sum/test.numInstances()) + " N: " + test.numInstances());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	public static JPanel graphSeries(ErrorModule preds, Instances traindata, String targetName, String xaxis){
		XYIntervalSeriesCollection xyDataset = new XYIntervalSeriesCollection();
	    XYIntervalSeries predictions = new XYIntervalSeries(targetName+" predictions", false, false);
	    XYIntervalSeries actual = new XYIntervalSeries(targetName, false, false);
	    xyDataset.addSeries(predictions);
	    xyDataset.addSeries(actual);
	    
	    ValueAxis timeAxis = null;
	    boolean timeAxisIsDate = false;
	    int timeIndex = -1;
	    if (traindata.attribute(xaxis).isDate()) {
	      timeAxis = new DateAxis("");
          timeAxisIsDate = true;
          timeIndex = traindata.attribute(xaxis).index();
        }
	    if (timeAxis == null) {
	        timeAxis = new NumberAxis("");
	        ((NumberAxis) timeAxis).setAutoRangeIncludesZero(false);
	    }
	    NumberAxis valueAxis = new NumberAxis("");
	    valueAxis.setAutoRangeIncludesZero(false);
	    List<NumericPrediction> predsForTargetAtI = preds.getPredictionsForTarget(targetName);
	    double x, y;
	    Attribute target= traindata.attribute(targetName);
	    for (int i = 0; i < traindata.numInstances(); i++) {
	    	if (timeAxisIsDate) {
	    		x = traindata.instance(i).value(timeIndex);
	    	} else {
	    		x = i;
	    	}
	    	//y=predsForTargetAtI.get(i).predicted();
	    	y=traindata.get(i).value(target);
			actual.add(x, x, x, y, y, y);
		}
	    int offset= traindata.numInstances() - predsForTargetAtI.size();
	    for (int i = 0; i < predsForTargetAtI.size(); i++) {
	    	if (timeAxisIsDate) {
	    		x = traindata.get(offset+i).value(timeIndex);
	    	} else {
	    		x = i;
	    	}
	    	y=predsForTargetAtI.get(i).predicted();
	    	predictions.add(x, x, x, y, y, y);
		}
		
	    XYErrorRenderer renderer = new XYErrorRenderer();
	    renderer.setBaseLinesVisible(true);
	    // renderer.setShapesFilled(true);
	    XYPlot plot = new XYPlot(xyDataset, timeAxis, valueAxis, renderer);
	    JFreeChart chart = new JFreeChart("Time series " + traindata.relationName(), JFreeChart.DEFAULT_TITLE_FONT,
	        plot, true);
	    chart.setBackgroundPaint(java.awt.Color.white);
	    TextTitle chartTitle = chart.getTitle();
	    String fontName = chartTitle.getFont().getFontName();
	    java.awt.Font newFont = new java.awt.Font(fontName, java.awt.Font.PLAIN, 12);
	    chartTitle.setFont(newFont);
	    ChartPanel result = new ChartPanel(chart, false, true, true, true, false);
	    return result;
	}
}

