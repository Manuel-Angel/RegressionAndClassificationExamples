package org.manuel.examples.RegressionAndClassificationExamples.weka;

import java.io.PrintStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.NumericPrediction;
import weka.classifiers.timeseries.AbstractForecaster;
import weka.classifiers.timeseries.TSForecaster;
import weka.classifiers.timeseries.WekaForecaster;
import weka.classifiers.timeseries.core.OverlayForecaster;
import weka.classifiers.timeseries.core.TSLagMaker;
import weka.classifiers.timeseries.core.TSLagUser;
import weka.classifiers.timeseries.eval.ErrorModule;
import weka.classifiers.timeseries.eval.RAEModule;
import weka.classifiers.timeseries.eval.RRSEModule;
import weka.classifiers.timeseries.eval.TSEvalModule;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class TimeSeries extends AbstractClassifier implements TSForecaster{
	private Classifier baseLearner;
	public WekaForecaster forecaster;
	public Instances instances;
	public Instances primeData;

	private String timestamp;
	private String fieldsToForecast;
	private PrintStream progress;
	private List<List<NumericPrediction>> forecastValues;
	private int m_primeWindowSize = 24;
	private int m_horizon = 24;
	public TimeSeries() {
		forecaster = new WekaForecaster();
		forecaster.getTSLagMaker().setMinLag(1);
		forecaster.getTSLagMaker().setMaxLag(24);
		progress = System.out;
	}

	public void main(String[] args) {
		// new forecaster
		try {
			Instances wine = Util.instancesFromFile("weather-10.arff");// Util.readXML("weather-10.xml");
			// set the targets we want to forecast. This method calls
			// setFieldsToLag() on the lag maker object for us

			// forecaster.setFieldsToForecast("windspeed");

			// default underlying classifier is SMOreg (SVM) - we'll use
			// gaussian processes for regression instead
			// baseLearner = (Classifier) new
			// weka.classifiers.functions.LinearRegression();
			// forecaster.setBaseForecaster(baseLearner);

			// forecaster.getTSLagMaker().setTimeStampField("date"); // date
			// time stamp

			// forecaster.getTSLagMaker().setMinLag(1);
			// forecaster.getTSLagMaker().setMaxLag(24); // hourly data

			// add a month of the year indicator field
			// forecaster.getTSLagMaker().setAddMonthOfYear(true);

			// add a quarter of the year indicator field
			// forecaster.getTSLagMaker().setAddQuarterOfYear(true);

			// build the model
			// forecaster.buildForecaster(wine, System.out);

			// prime the forecaster with enough recent historical data
			// to cover up to the maximum lag. In our case, we could just supply
			// the 12 most recent historical instances, as this covers our
			// maximum
			// lag period
			forecaster.primeForecaster(wine);

			// forecast for 12 units (months) beyond the end of the
			// training data
			List<List<NumericPrediction>> forecast;

			forecast = forecaster.forecast(24, System.out);

			// output the predictions. Outer list is over the steps; inner list
			// is over
			// the targets
			for (int i = 0; i < forecast.size(); i++) {
				List<NumericPrediction> predsAtStep = forecast.get(i);
				for (int j = 0; j < predsAtStep.size(); j++) {
					NumericPrediction predForTarget = predsAtStep.get(j);
					System.out.print("" + predForTarget.predicted() + " ");
				}
				System.out.println();
			}

			// we can continue to use the trained forecaster for further
			// forecasting
			// by priming with the most recent historical data (as it becomes
			// available).
			// At some stage it becomes prudent to re-build the model using
			// current
			// historical data.

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void runForecaster() {
		forecaster.runForecaster(forecaster, new String[] { "-S", "0", "-R", "1.0E-8", "-num-decimal-places", "4",
				"-prime", "" + instances.numInstances(), "-F", "fieldsToForecast" });
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		instances = data;
		long timestart = System.currentTimeMillis();
		if (getBaseLearner() == null) {
			setBaseLearner((Classifier) new weka.classifiers.functions.LinearRegression());
		}
		forecaster.setBaseForecaster(getBaseLearner());
		if (getFieldsToForecast() == null) {
			Attribute classAt = instances.classAttribute();
			setFieldsToForecast(classAt.name());
		}
		forecaster.setFieldsToForecast(getFieldsToForecast());
		if (getTimestamp() == null) {
			for (int i = 0; i < instances.numAttributes(); i++) {
				if (instances.attribute(i).type() == Attribute.DATE) {
					setTimestamp(instances.attribute(i).name());
					break;
				}
			}
		}
		forecaster.getTSLagMaker().setTimeStampField(getTimestamp());
		if (m_Debug) {
			System.out.println("building forecaster...");
			forecaster.buildForecaster(instances, progress);
			long act = System.currentTimeMillis();
			System.out.println("build in " + (act - timestart));
			System.out.println("prime forecast...");
			timestart = System.currentTimeMillis();
		} else {
			forecaster.buildForecaster(instances);
		}
		forecaster.primeForecaster(instances);
		if (m_Debug) {
			System.out.println("time to prime forcaster to run: " + (System.currentTimeMillis() - timestart));
		}
	}

	@Override
	public double classifyInstance(Instance instance) throws Exception {
		int steps;
		Attribute att = instances.attribute(timestamp);
		long time = (long) instance.value(att);

		Instance ult = instances.lastInstance();// instances.get(instances.numInstances()-1);
		long timeUlt = (long) ult.value(att);
		steps = (int) ((time - timeUlt) / (1000 * 60 * 60)) - 1; // -1 because
																	// the
																	// forecast
																	// values
																	// start
																	// from t+1
		if (m_Debug) {
			Date dateInst = new Date(time);
			Date dateUlt = new Date(timeUlt);

			System.out.println("ultima fecha        " + timeUlt + " " + att.formatDate(timeUlt)); // dateUlt.toString());
			System.out.println("fecha a pronosticar " + time + " " + att.formatDate(time)); // dateInst.toString());
			System.out.println("delta: " + this.getTSLagMaker().getDeltaTime());
			System.out.println("steps: " + steps);
		}

		if (forecastValues == null || steps > forecastValues.size()) {
			if (m_Debug) {
				System.out.println("Forecasting...");
				forecastValues = forecaster.forecast(Math.max(24, steps), progress);
				System.out.println("prediccion: ");
				for (int i = 0; i < forecastValues.size(); i++) {
					List<NumericPrediction> ac = forecastValues.get(i);
					System.out.println(ac.get(0).actual() + " " + ac.get(0).predicted());
				}
			} else {
				forecastValues = forecaster.forecast(Math.max(24, steps));
			}
		}
		NumericPrediction predForTarget = forecastValues.get(steps).get(0);
		return predForTarget.predicted();
	}

	public List<ErrorModule> classifyInstance(Instances m_trainingData, Instances m_testData) throws Exception {
		// set up training set prediction and eval modules
		List<ErrorModule> m_predictionsForTestData;
		m_predictionsForTestData = new ArrayList<ErrorModule>();
		Map<String, List<TSEvalModule>> m_metricsForTestData;
		m_metricsForTestData = new HashMap<String, List<TSEvalModule>>();

		TSForecaster forecaster = (TSForecaster) this.forecaster;
		setupEvalModules(m_predictionsForTestData, m_metricsForTestData,
				AbstractForecaster.stringToList(forecaster.getFieldsToForecast()));

		Instances primeData = null;
		Instances rebuildData = null;

		boolean m_rebuildModelAfterEachTestForecastStep = false;
		//m_primeWindowSize = 24; m_horizon = 24;

		forecaster.buildForecaster(m_trainingData);
		
		if (m_trainingData != null) { // if innecesario
			primeData = new Instances(m_trainingData, 0);
			if (forecaster instanceof TSLagUser) {
				// initialize the artificial time stamp value (if in use)
				if (((TSLagUser) forecaster).getTSLagMaker().isUsingAnArtificialTimeIndex()) {
					((TSLagUser) forecaster).getTSLagMaker().setArtificialTimeStartValue(m_trainingData.numInstances());
				}
			}
			if (m_rebuildModelAfterEachTestForecastStep) {
				rebuildData = new Instances(m_trainingData);
			}
		} else {
			primeData = new Instances(m_testData, 0);
		}

		int predictionOffsetForTestData = 0;
		// use the last primeWindowSize instances from the training
		// data to prime with
		predictionOffsetForTestData = 0;
		primeData = new Instances(m_trainingData, (m_trainingData.numInstances() - m_primeWindowSize),
				m_primeWindowSize);
		System.out.println(primeData);
		for (int i = predictionOffsetForTestData; i < m_testData.numInstances(); i++) {
			Instance current = m_testData.instance(i);
			if (m_primeWindowSize > 0) {
				forecaster.primeForecaster(primeData);
			}
			List<List<NumericPrediction>> forecast = null;

			forecast = forecaster.forecast(m_horizon, progress);

			List<NumericPrediction> predsAtStep = forecast.get(0);
			
			for (int j = 0; j < predsAtStep.size(); j++) {
				System.out.print(predsAtStep.get(j).predicted()+",");
			}
			System.out.println();
			
			updateEvalModules(m_predictionsForTestData, m_metricsForTestData, forecast, i, m_testData);

			// remove the oldest prime instance and add this one
			if (m_primeWindowSize > 0 && current != null) {
				primeData.remove(0);
				primeData.add(current);
				primeData.compactify();
			}

			// add the current instance to the training data and rebuild the
			// forecaster (if requested and if there is training data).
			if (m_rebuildModelAfterEachTestForecastStep && rebuildData != null) {
				rebuildData.add(current);
				forecaster.buildForecaster(rebuildData);
			}
		}

		ErrorModule predsForStep = m_predictionsForTestData.get(1 -1);
	    List<NumericPrediction> preds = predsForStep
	      .getPredictionsForTarget(getFieldsToForecast());
	    
	    for (int i = 0; i < preds.size(); i++) {
	    	NumericPrediction pred = preds.get(i);
	    	System.out.printf("actual %10f predicted %10f\n", pred.actual(), pred.predicted());
		}
		return m_predictionsForTestData;
	}

	private void updateEvalModules(List<ErrorModule> predHolders, Map<String, List<TSEvalModule>> evalHolders,
			List<List<NumericPrediction>> predsForSteps, int currentInstanceNum, Instances toPredict) throws Exception {

		// errors first
		for (int i = 0; i < predsForSteps.size(); i++) {
			// when using overlay data there will only be as many predictions as
			// there
			// are
			// instances to predict
			if (i < predsForSteps.size()) {
				List<NumericPrediction> predsForStepI = predsForSteps.get(i);
				if (currentInstanceNum + i < toPredict.numInstances()) {
					predHolders.get(i).evaluateForInstance(predsForStepI, toPredict.instance(currentInstanceNum + i));
				} else {
					predHolders.get(i).evaluateForInstance(predsForStepI, null);
				}
			}
		}
	}

	private void setupEvalModules(List<ErrorModule> predHolders, Map<String, List<TSEvalModule>> evalHolders,
			List<String> fieldsToForecast) {
		for (int i = 0; i < 25; i++) {
			ErrorModule e = new ErrorModule();
			e.setTargetFields(fieldsToForecast);
			predHolders.add(e);
		}
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		return new double[] { classifyInstance(instance) };
	}

	public TSLagMaker getTSLagMaker() {
		return forecaster.getTSLagMaker();
	}

	public String getTimestamp() {
		return timestamp;
	}

	public void setTimestamp(String timestamp) {
		try {
			forecaster.getTSLagMaker().setTimeStampField(timestamp);
		} catch (Exception e) {
			e.printStackTrace();
		}
		this.timestamp = timestamp;
	}

	public String getFieldsToForecast() {
		return fieldsToForecast;
	}

	public void setFieldsToForecast(String fieldsToForecast) {
		try {
			forecaster.setFieldsToForecast(fieldsToForecast);
		} catch (Exception e) {
			e.printStackTrace();
		}
		this.fieldsToForecast = fieldsToForecast;
	}

	public Classifier getBaseLearner() {
		return baseLearner;
	}

	public void setBaseLearner(Classifier baseLearner) {
		forecaster.setBaseForecaster(baseLearner);
		this.baseLearner = baseLearner;
	}

	
	
	
	@Override
	public String getAlgorithmName() {
		return baseLearner.getClass().getName();
	}

	@Override
	public void reset() {
		this.forecaster.reset();
	}

	@Override
	public void buildForecaster(Instances insts, PrintStream... progress) throws Exception {
		//buildClassifier(insts);
		throw new RuntimeException("buildForecaster not implemented yet");
	}

	@Override
	public void primeForecaster(Instances insts) throws Exception {
		throw new RuntimeException("primeForecaster not implemented yet");
		//this.forecaster.primeForecaster(insts);
	}

	@Override
	public List<List<NumericPrediction>> forecast(int numSteps, PrintStream... progress) throws Exception {
		// TODO Auto-generated method stub
		throw new RuntimeException("forecast not implemented yet");
		//return this.forecaster.forecast(numSteps, progress);
	}

	@Override
	public void runForecaster(TSForecaster forecaster, String[] options) {
		// TODO Auto-generated method stub
		throw new RuntimeException("not implemented yet");
	}

	public List<ErrorModule> forecast(Instances train, Instances test) throws Exception {
		List<ErrorModule> preds= new ArrayList<>(1);
		buildClassifier(train);
		forecastValues = forecaster.forecast(test.numInstances());
		
		ErrorModule em= new ErrorModule();
		List<String> f=new ArrayList<String>();
		f.add(fieldsToForecast);
		em.setTargetFields(f);
		for (int i = 0; i < test.numInstances(); i++) {
			List<NumericPrediction> predsForStepI = forecastValues.get(i);	
			em.evaluateForInstance(predsForStepI, test.get(i));
		}
		preds.add(em);
		return preds;
	}

	public int getHorizon() {
		return m_horizon;
	}

	public void setHorizon(int m_horizon) {
		this.m_horizon = m_horizon;
	}

	public int getPrimeWindowSize() {
		return m_primeWindowSize;
	}

	public void setPrimeWindowSize(int m_primeWindowSize) {
		this.m_primeWindowSize = m_primeWindowSize;
	}
}
