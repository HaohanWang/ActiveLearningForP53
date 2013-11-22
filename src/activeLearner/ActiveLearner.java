package activeLearner;

import weka.classifiers.functions.SMO;
import learnerHelper.DataManager;

public class ActiveLearner {
	DataManager dm = null;
	SMO classifier = null;
	
	public ActiveLearner(){
		dm = DataManager.getInstance();
		classifier = new SMO();
	}
	
	
}
