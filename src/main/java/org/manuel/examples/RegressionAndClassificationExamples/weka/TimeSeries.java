package org.manuel.examples.RegressionAndClassificationExamples.weka;

import java.io.PrintStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.NumericPrediction;
import weka.classifiers.timeseries.TSForecaster;
import weka.classifiers.timeseries.WekaForecaster;
import weka.classifiers.timeseries.core.TSLagMaker;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class TimeSeries extends AbstractClassifier {
	public Classifier baseLearner;
	public WekaForecaster forecaster;
	public Instances instances;
	public Instances primeData;
	
	private String timestamp;
	private String fieldsToForecast;
	private PrintStream progress;
	private List<List<NumericPrediction>> forecastValues;
	public TimeSeries(){
		forecaster = new WekaForecaster();
		forecaster.getTSLagMaker().setMinLag(1);
		forecaster.getTSLagMaker().setMaxLag(24);
		progress = System.out;
	}
	
	public  void main(String[] args) {
		// new forecaster
		try {
			Instances wine = Util.instancesFromFile("weather-10.arff");//Util.readXML("weather-10.xml");
			// set the targets we want to forecast. This method calls
			// setFieldsToLag() on the lag maker object for us
			
			//forecaster.setFieldsToForecast("windspeed");

			// default underlying classifier is SMOreg (SVM) - we'll use
			// gaussian processes for regression instead
			//baseLearner = (Classifier) new weka.classifiers.functions.LinearRegression();
			//forecaster.setBaseForecaster(baseLearner);

			//forecaster.getTSLagMaker().setTimeStampField("date"); // date time stamp
			
			//forecaster.getTSLagMaker().setMinLag(1);
			//forecaster.getTSLagMaker().setMaxLag(24); // hourly data

			// add a month of the year indicator field
			//forecaster.getTSLagMaker().setAddMonthOfYear(true);

			// add a quarter of the year indicator field
			//forecaster.getTSLagMaker().setAddQuarterOfYear(true);

			// build the model
			//forecaster.buildForecaster(wine, System.out);

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
	public void runForecaster(){
		forecaster.runForecaster(forecaster, new String[]{"-S", "0", "-R", "1.0E-8", "-num-decimal-places", 
				"4", "-prime", "" + instances.numInstances(), "-F","fieldsToForecast"});
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		instances= data;
		long timestart= System.currentTimeMillis();
		if(baseLearner==null){
			baseLearner= (Classifier) new weka.classifiers.functions.LinearRegression();
		}
		forecaster.setBaseForecaster(baseLearner);
		if(getFieldsToForecast()==null){
			Attribute classAt=instances.classAttribute();
			setFieldsToForecast(classAt.name());
		}
		forecaster.setFieldsToForecast(getFieldsToForecast());
		if(getTimestamp()==null){
			for (int i = 0; i < instances.numAttributes(); i++) {
				if(instances.attribute(i).type() == Attribute.DATE){
					setTimestamp(instances.attribute(i).name());
					break;
				}
			}
		}
		forecaster.getTSLagMaker().setTimeStampField(getTimestamp());
		if(m_Debug){
			System.out.println("building forecaster...");
			forecaster.buildForecaster(instances, progress);
			long act= System.currentTimeMillis();
			System.out.println("build in "+ (act-timestart));
			System.out.println("prime forecast...");
			timestart= System.currentTimeMillis();
		}else {
			forecaster.buildForecaster(instances);
		}
		forecaster.primeForecaster(instances);
		if(m_Debug){
			System.out.println("time to prime forcaster to run: "+ (System.currentTimeMillis() - timestart));
		}
	}
	@Override
	public double classifyInstance(Instance instance) throws Exception {
		int steps;
		Attribute att= instances.attribute(timestamp);
		long time = (long)instance.value(att);
		
		Instance ult=instances.lastInstance();// instances.get(instances.numInstances()-1);
		long timeUlt = (long)ult.value(att);
		steps=(int) ((time - timeUlt)/(1000*60*60)) - 1; //-1 because the forecast values start from t+1
		if(m_Debug){
			Date dateInst= new Date(time);
			Date dateUlt= new Date(timeUlt);
			
			System.out.println("ultima fecha        "+timeUlt + " "+ att.formatDate(timeUlt)); //dateUlt.toString());
			System.out.println("fecha a pronosticar "+time + " " + att.formatDate(time));  //dateInst.toString());
			System.out.println("delta: " + this.getTSLagMaker().getDeltaTime());
			System.out.println("steps: " + steps);
		}
		
		if(forecastValues==null || steps > forecastValues.size()){
			if(m_Debug){
				System.out.println("Forecasting...");
				forecastValues = forecaster.forecast(Math.max(24, steps), progress);
			}else{
				forecastValues = forecaster.forecast(Math.max(24, steps));
			}
		}
		NumericPrediction predForTarget= forecastValues.get(steps).get(0);
		return predForTarget.predicted();
	}
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		return new double[]{classifyInstance(instance)};
	}
	public TSLagMaker getTSLagMaker(){
		return forecaster.getTSLagMaker();
	}
	public String getTimestamp() {
		return timestamp;
	}
	public void setTimestamp(String timestamp) {
		this.timestamp = timestamp;
	}
	public String getFieldsToForecast() {
		return fieldsToForecast;
	}
	public void setFieldsToForecast(String fieldsToForecast) {
		this.fieldsToForecast = fieldsToForecast;
	}

}
