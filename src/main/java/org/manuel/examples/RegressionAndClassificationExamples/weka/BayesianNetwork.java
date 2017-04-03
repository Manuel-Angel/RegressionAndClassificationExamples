package org.manuel.examples.RegressionAndClassificationExamples.weka;

import java.awt.BorderLayout;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import javax.swing.JFrame;
import javax.swing.text.LabelView;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.BayesNetEstimator;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.classifiers.bayes.net.search.local.K2;
import weka.classifiers.evaluation.NumericPrediction;
import weka.classifiers.timeseries.eval.ErrorModule;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.graphvisualizer.BIFFormatException;
import weka.gui.graphvisualizer.GraphVisualizer;

public class BayesianNetwork extends AbstractClassifier{
	private BayesNet red= new BayesNet();
	private BayesNetEstimator estimator;
	private SearchAlgorithm search;
	private double labelValues[];
	private HashMap<Integer, LabelData> labelData= new HashMap<>();
	private Instances data;
	/**
	 * Filters a dataset making it nominal (with weka.filters.unsupervised.attribute.Discretize 
	 * filter) and removing the date attribute.
	 * @param data
	 * @return
	 */
	public Instances filterInstancesForBayesNet(Instances data){
		//remove date attribute
		Remove removeFilter = new Remove();
		int attInd = data.attribute("date").index();
		//removeFilter.setAttributeIndicesArray(new int[]{attInd});
		//removeFilter.setAttributeIndices("2");
		String[] options = new String[2];
		options[0] = "-R";
		options[1] = "" + (attInd+1);
		
		try {
			removeFilter.setOptions(options);
			removeFilter.setInputFormat(data);
		} catch (Exception e2) {
			e2.printStackTrace();
			return null;
		}
		
		Instances firstFilter;
		try {
			firstFilter = Filter.useFilter(data, removeFilter);
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
		//Instances firstFilter= removeFilter.getOutputFormat();
			//firstFilter.setClassIndex(data.classIndex());
		
		//System.out.println("class index: " + firstFilter.attribute(firstFilter.classIndex()));
		
		//firstFilter.deleteAttributeType(Attribute.DATE);
		
		System.out.println("first filter: \n" + firstFilter);
		
		/*for (int i = 0; i < data.numInstances(); i++) {
			removeFilter.input(data.get(i));
			data.get(i).deleteAttributeAt(attInd);
			firstFilter.add(removeFilter.output());
		}*/
		
		firstFilter = numericToNominal(firstFilter, firstFilter.classIndex(), 36);
		System.out.println("second filter: \n" + firstFilter);
		
		//dizcretize data
		Discretize filtro= new Discretize();
    	try {
			filtro.setInputFormat(firstFilter);
		} catch (Exception e1) {
			e1.printStackTrace();
			return null;
		}
    	
    	//TODO cambiar esto por Filter.useFilter
    	
    	//filtro.setAttributeIndicesArray(new int[]{0,1,2,3,4});
    	//filtro.setBinRangePrecision(10);
    	//filtro.setBins(20);
    	for (int i = 0; i < firstFilter.numInstances(); i++) {
			filtro.input(firstFilter.get(i));
		}

    	filtro.batchFinished();
    	Instances newData= filtro.getOutputFormat();
    	for (int i = 0; i < firstFilter.numInstances(); i++) {
			newData.add(filtro.output());
		}
		return newData;
	}
	protected Instances numericToNominal(Instances data, int att, int bins){
		double maxi=Double.NEGATIVE_INFINITY, mini= Double.MAX_VALUE;
		for (int i = 0; i < data.numInstances(); i++) {
			Instance act=data.get(i);
			maxi= Math.max(act.value(att), maxi);
			mini= Math.min(act.value(att), mini);
		}
		double separacion = (maxi - mini)/ bins;
		labelValues = new double[bins+1];
		System.out.println("min "+ mini + ", max "+ maxi +" separacion "+separacion);
		labelData.put(att, new LabelData(mini, separacion, bins));
		ArrayList<Attribute> attribs= new ArrayList<>(data.numAttributes());
		for (int i = 0; i < data.numAttributes(); i++) {
			if(i==att){
				ArrayList<String> labels = new ArrayList<>(bins+1);
				for (int j = 0; j < bins; j++) {
					labels.add("["+(mini + separacion*(j))+"-" + (mini + separacion*(j+1)) +")");
					labelValues[j]= mini + separacion*(j+0.5);
				}
				labelValues[bins] = mini + separacion*(bins+0.5);
				labels.add("["+maxi +"-inf)");
				System.out.println("labels: " +labels);
				attribs.add(new Attribute(data.attribute(att).name(), labels));
			} else{
				attribs.add(data.attribute(i)); //doesn't do deep copy
			}
		}
		Instances newInst= new Instances(data.relationName(),attribs,data.numInstances());
		newInst.setClassIndex(data.classIndex());
		for (int c = 0; c < data.numInstances(); c++) {
			Instance act=data.get(c);
			double values[] =  new double[data.numAttributes()];
			for (int i = 0; i < data.numAttributes(); i++) {
				if(i==att){
					values[i]= ((act.value(i) - mini) /separacion);
					//values[i]= Math.max(0, values[i]);
					values[i]= Math.min(bins, values[i]);
					/*System.out.println("Valor " + act.value(i)+ " puesto en "
							+ newInst.attribute(i).value((int)values[i]) + " (" +values[i]+")" );*/
				} else{
					values[i]= act.value(i);
				}
			}
			newInst.add(Util.instance(values, newInst));
		}
		return newInst;
	}
	public static void visualizeBayesNet(String graph){
		GraphVisualizer gv = new GraphVisualizer();
		try {
			gv.readBIF(graph);
		} catch (BIFFormatException e) {
			e.printStackTrace();
			return ;
		}
		JFrame jf = new JFrame("BayesNet graph");
		jf.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		jf.setSize(800, 600);
		jf.getContentPane().setLayout(new BorderLayout());
		jf.getContentPane().add(gv, BorderLayout.CENTER);
		jf.setVisible(true);
		// layout graph
		gv.layoutGraph();
	}
	public Instances discretizeData(Instances data){
		Remove removeFilter = new Remove();
		int attInd = data.attribute("date").index();
		//removeFilter.setAttributeIndicesArray(new int[]{attInd});
		//removeFilter.setAttributeIndices("2");
		String[] options = new String[2];
		options[0] = "-R";
		options[1] = "" + (attInd+1);
		
		try {
			removeFilter.setOptions(options);
			removeFilter.setInputFormat(data);
		} catch (Exception e2) {
			e2.printStackTrace();
			return null;
		}
		Instances filteredData;
		try {
			filteredData = Filter.useFilter(data, removeFilter);
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
		//TODO cambiar esto por un metodo mas rapido
		int idx= data.attribute("temp").index();
		filteredData = numericToNominal(filteredData, idx, 25);
		idx= data.attribute("cloudcover").index();
		filteredData = numericToNominal(filteredData, idx, 10);
		idx= data.attribute("windspeed").index();
		filteredData = numericToNominal(filteredData, idx, 15);
		idx= data.attribute("winddir").index();
		filteredData = numericToNominal(filteredData, idx, 36);
		idx= data.attribute("kw").index(); //this should be the last to save the labelValues of the target values
		filteredData = numericToNominal(filteredData, idx, 20);
		return filteredData;
	}
	@Override
	public void buildClassifier(Instances data) throws Exception {
		Instances filteredData= discretizeData(data);
		if(estimator==null){
			estimator=new SimpleEstimator();
		}
		if(search==null){
			search=new K2();
		}
		this.data = filteredData; 
		red.setEstimator(estimator);
    	red.setSearchAlgorithm(search);
    	red.buildClassifier(filteredData);
	}
	@Override
	public double classifyInstance(Instance instance) throws Exception {
		int idx= instance.dataset().attribute("date").index();
		instance.setDataset(null);
		instance.deleteAttributeAt(idx);
		
		discretize(instance, "temp");
		discretize(instance, "cloudcover");
		discretize(instance, "windspeed");
		discretize(instance, "winddir");
		discretize(instance, "kw");
		
		instance.setDataset(data);
		double distr[]=red.distributionForInstance(instance);
		double r=0;
		System.out.println("distribution for instance" + distr.length+ "  values" + labelValues.length);
		for (int i = 0; i < distr.length; i++) {
			r+= labelValues[i]*distr[i];
		}
		return r;
	}
	@Override
	protected Object clone() throws CloneNotSupportedException {
		return super.clone();
	}
	public List<ErrorModule> trainAndTest(Instances train, Instances test) throws Exception {
		List<ErrorModule> preds= new ArrayList<>(1);
		buildClassifier(train);
		String fieldsToForecast= train.classAttribute().name();
		ErrorModule em= new ErrorModule();
		List<String> f=new ArrayList<String>();
		f.add(fieldsToForecast);
		em.setTargetFields(f);
		Attribute att= test.attribute(fieldsToForecast);
		for (int i = 0; i < test.numInstances(); i++) {
			double predicted= classifyInstance(test.get(i));
			List<NumericPrediction> predsForStepI = new ArrayList<>(1);
			NumericPrediction pred= new NumericPrediction(test.get(i).value(att), predicted);
			predsForStepI.add(pred);
			em.evaluateForInstance(predsForStepI, test.get(i));
		}
		preds.add(em);
		return preds;
	}
	public void discretize(Instance instance, String attrib){
		int idx= data.attribute(attrib).index();
		LabelData lb= labelData.get(idx);
		double value= (instance.value(idx)-lb.mini) / lb.separacion;
		value= Math.max(value, 0);
		value= Math.min(value, lb.bins);
		instance.setValue(idx, value);
	}
	class LabelData{
		double mini;
		double separacion;
		int bins;
		LabelData(double min, double sep, int bins){
			mini=min;
			separacion=sep;
			this.bins= bins;
		}
	}
}
