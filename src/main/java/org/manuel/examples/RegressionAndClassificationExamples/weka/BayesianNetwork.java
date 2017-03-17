package org.manuel.examples.RegressionAndClassificationExamples.weka;

import java.awt.BorderLayout;
import java.util.ArrayList;
import java.util.HashSet;

import javax.swing.JFrame;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.graphvisualizer.BIFFormatException;
import weka.gui.graphvisualizer.GraphVisualizer;

public class BayesianNetwork {
	/**
	 * Filters a dataset making it nominal (with weka.filters.unsupervised.attribute.Discretize 
	 * filter) and removing the date attribute.
	 * @param data
	 * @return
	 */
	public static Instances filterInstancesForBayesNet(Instances data){
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
		
		firstFilter = numericToNominal(firstFilter, firstFilter.classIndex());
		System.out.println("second filter: \n" + firstFilter);
		
		//dizcretize data
		Discretize filtro= new Discretize();
    	try {
			filtro.setInputFormat(firstFilter);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
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
	protected static Instances numericToNominal(Instances data, int att){
		double maxi=Double.NEGATIVE_INFINITY, mini= Double.MAX_VALUE;
		HashSet<Integer> set= new HashSet<>();
		for (int i = 0; i < data.numInstances(); i++) {
			Instance act=data.get(i);
			maxi= Math.max(act.value(att), maxi);
			mini= Math.min(act.value(att), mini);
			int val= (int)act.value(att);
			set.add(val);
		}
		
		int bins = Math.min(set.size(), 36);
		double separacion = (maxi - mini)/ bins;
		System.out.println("min "+ mini + ", max "+ maxi +" separacion "+separacion+ " set" + set);
		ArrayList<Attribute> attribs= new ArrayList<>(data.numAttributes() - 1);
		for (int i = 0; i < data.numAttributes(); i++) {
			if(i==att){
				ArrayList<String> labels = new ArrayList<>(bins+1);
				for (int j = 0; j < bins; j++) {
					labels.add("["+(mini + separacion*(j))+"-" + (mini + separacion*(j+1)) +")");
				}
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
}
