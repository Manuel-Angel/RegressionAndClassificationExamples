package org.manuel.examples.RegressionAndClassificationExamples.weka;

import java.awt.BorderLayout;
import java.awt.Color;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;

import javax.swing.JFrame;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.local.K2;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
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
	
}
