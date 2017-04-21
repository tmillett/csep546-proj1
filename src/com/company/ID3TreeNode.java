package com.company;

import com.company.AttrInfo;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by tmillett on 4/10/17.
 */
public class ID3TreeNode {

    private ID3TreeNode parent;
    private Integer confidenceLevel;
    protected Attribute attribute;
    private Map<Double, ID3TreeNode> children;
    private Double terminatedClassValue;
    private Double parentAttrValue;

    public ID3TreeNode(ID3TreeNode parent, Integer confidenceLevel) {
        this.parent = parent;
        this.confidenceLevel = confidenceLevel;
        this.children = new HashMap<>();
        this.terminatedClassValue = null;
    }

    public void setAttribute(Attribute attribute) {
        this.attribute = attribute;
    }

    public void setChildForAttributeValue(Double attributeValue, ID3TreeNode node) {
        this.children.put(attributeValue, node);
    }

    private Attribute findBestAttribute(Instances data) {
        // C_i = percent of ith class value to total number of all classes
        // Calculate the entropy: E(S) = Sum(-p(C_i)log_2(C_i))
        double entropy = getTableEntropy(data);

        // Create structure to aid in calculating gain
        Map<Attribute, GainInfo> instanceRowsByClass = getAttributeGainInfoMap(data);

        Attribute attributeWithHighestGain = calculateGain(entropy, instanceRowsByClass);

        return attributeWithHighestGain;
    }

    private double getTableEntropy(Instances data) {
        int totalNumOfRows = data.numInstances();
        AttributeStats classStats = data.attributeStats(data.classIndex());

        double entropy = 0.0;
        for (int i = 0; i < classStats.nominalCounts.length; i++) {
            int count = classStats.nominalCounts[i];
            if (count != totalNumOfRows && count > 0) {
                double percentage = (double) count / (double) totalNumOfRows;
                entropy -= percentage * (Math.log10(percentage) / Math.log10(2));
            }
        }
        return entropy;
    }

    private Attribute calculateGain(double entropy, Map<Attribute, GainInfo> instanceRowsByClass) {
        Attribute attributeWithHighestGain = null;
        double highestGain = -1.0;
        for (Attribute attr : instanceRowsByClass.keySet()) {
            double currentGain = instanceRowsByClass.get(attr).gain(entropy);

            if (currentGain > highestGain) {
                attributeWithHighestGain = attr;
                highestGain = currentGain;

            }
        }

        return attributeWithHighestGain;
    }

    private Map<Attribute, GainInfo> getAttributeGainInfoMap(Instances data) {
        /*
            Calculate the gain for each attribute
            Data Structure is following HashMap
            Key is attribute class
            Value is another HashMap with Key being specific enumerations of the attribute,
            Value being array of size [numClassValues] and counts of each type
            eg {{M,F} ->{ M-> [2,3,1], F -> [4, 0, 1]}}
        */

        Map<Attribute, GainInfo> instanceRowsByClass = new HashMap<>(data.size());
        for (int i = 0; i < data.size(); i++) {
            Instance instance = data.instance(i);
            double classValue = instance.classValue();
            for (int j = 0; j < instance.numValues(); j++) {

                if (j == instance.classIndex()) continue;

                Attribute attr = instance.attribute(j);
                Double attrValue = instance.value(j);

                if (attr != null) {
                    GainInfo classCountPerAttrType = instanceRowsByClass.get(attr);
                    if (classCountPerAttrType == null) {
                        classCountPerAttrType = new GainInfo(attr, instance.classAttribute());
                        instanceRowsByClass.put(attr, classCountPerAttrType);
                    }

                    classCountPerAttrType.addInstance(attrValue, classValue);
                }
            }
        }
        return instanceRowsByClass;
    }

    public void train(Instances data) {

        // Find the attribute to split on to figure out the most information gain
        Attribute root = this.findBestAttribute(data);

        // Find the index of the attribute class
        this.setAttribute(root);
        int attributeIndex = -1;
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attr = data.attribute(i);
            if (attr == root) {
                attributeIndex = i;
                break;
            }
        }

        // Split all the instances into groups. The groups are split
        // by the value of the Attribute root
        // ex Attribute is Gender, subInstancesMap will be {M ->[I1,I3], F -> [I2,I4,I5]}

        // While we are at it, decide if all the subInstances in a group have that same class value
        // subInstancesShouldGrowMap will contain a a TRUE or FALSE for that attribute value
        // If FALSE, then subInstancesClassValueMap will contain that class value
        // If TRUE, then subInstancesShouldGrowMap
        Map<Double, Instances> subInstancesMap = new HashMap<>();
        Map<Double, Double> subInstancesClassValueMap = new HashMap<>();
        Map<Double, Boolean> subInstancesShouldGrowMap = new HashMap<>();

        for (int i = 0; i < data.size(); i++) {
            Instance instance = data.instance(i);
            Double attrValue = instance.value(root);
            double classValue = instance.classValue();


            Instances subInstances = subInstancesMap.get(attrValue);
            if (subInstances == null) {
                subInstances = new Instances(data, data.size());
                subInstancesMap.put(attrValue, subInstances);
            }
            subInstances.add(instance);

            Double existingClassValue = subInstancesClassValueMap.get(attrValue);
            if (existingClassValue == null) {
                existingClassValue = classValue;
                subInstancesClassValueMap.put(attrValue, existingClassValue);
            }

            if (existingClassValue != classValue) {
                subInstancesShouldGrowMap.put(attrValue, true);
            }
        }

        Boolean isSignificant = isBranchStatisticallySignificant(data, root, subInstancesMap);

        if (isSignificant) {

            // We have a chosen an attribute type and sorted the instances by the possible attribute values
            // We have discovered if all the class values for a given attribute value match
            // We are now ready to either grow the tree (because class values in a specific group do not match)
            // Or add a leaf node with a specific attribute value and a specific class value
            for (Double attrValue : subInstancesMap.keySet()) {

                Instances subInstances = subInstancesMap.get(attrValue);
                Boolean shouldGrowTree = subInstancesShouldGrowMap.get(attrValue);

                if (shouldGrowTree != null && shouldGrowTree == true) {
                    subInstances.deleteAttributeAt(attributeIndex);
                    ID3TreeNode childNode = new ID3TreeNode(this, this.confidenceLevel);
                    this.setChildForAttributeValue(attrValue, childNode);
                    childNode.train(subInstances);
                } else {
                    Double existingClassValue = subInstancesClassValueMap.get(attrValue);

                    ID3TreeLeaf leafNode = new ID3TreeLeaf(this, this.confidenceLevel);
                    leafNode.setAttribute(this.attribute);

                    leafNode.setClassValueForAttributeValue(attrValue, existingClassValue);
                    this.setChildForAttributeValue(attrValue, leafNode);
                }
            }
        } else {
            Map<Double, Integer> counts = new HashMap<>();
            countNumOutcomes(data, counts);
            Integer highestCount = Integer.MIN_VALUE;
            Double highestClassValue = null;
            for (Double classValue : counts.keySet()) {
                Integer count = counts.get(classValue);
                if (count > highestCount) {
                    highestCount = count;
                    highestClassValue = classValue;
                }
            }

            this.terminatedClassValue = highestClassValue;
        }
    }

    private Boolean isBranchStatisticallySignificant(Instances instances, Attribute attribute, Map<Double, Instances> subInstancesMap) {

        if (this.confidenceLevel == 2) {
            return true;
        }

//        Start off with entire table
//        choose an attribute attr1
//        get number of positives and negatives, total outcome for table -> totalPos, totalNeg, total
//        collect sum of loop -> sumChi
//        loop on each possible attr1Value -> attrValue
//          find number of positives and negatives for attrValue and total outcomes -> attrValuePos, attrValueNeg attrValueTotal
//          find expected positives for attrValue -> round(attrValueTotal * (totalPos/total)) -> expectedPos
//          find expected negatives for attrValue -> attrValueTotal - expectedPos -> expectedNeg
//          sumChi += ((attrValuePos-expectedPos)^2/expectedPos) + ((attrValueNeg-expectedNeg)^2/expectedNeg)

        // {classValue1 -> count, classValue2 -> count}
        Integer totalCount = instances.size();
        Map<Double, Integer> totalNumOfEachOutcome = new HashMap<>();
        countNumOutcomes(instances, totalNumOfEachOutcome);

        Map<Double, Map<Double, Integer>> numOfEachOutcome = new HashMap<>();
        for (Double attrValue: subInstancesMap.keySet()) {
            Instances subInstances = subInstancesMap.get(attrValue);
            Map<Double, Integer> counts = numOfEachOutcome.get(attrValue);
            if (counts == null) {
                counts = new HashMap<>();
                numOfEachOutcome.put(attrValue,counts);
            }

            countNumOutcomes(subInstances,counts);
        }

        Double subChiSquared = 0.0;
        for (Double attrValue: numOfEachOutcome.keySet()) {
            Map<Double, Integer> counts = numOfEachOutcome.get(attrValue);
            Integer actualAttrValueCount = 0;
            for (Double classValue: counts.keySet()) {
                actualAttrValueCount += counts.get(classValue);
            }

            for (Double classValue: counts.keySet()) {
                Integer totalCountForClassValue = totalNumOfEachOutcome.get(classValue);
                if (totalCountForClassValue > 0) {
                    Double expectedAttrValueCountForClassValue = actualAttrValueCount.doubleValue() * (totalCountForClassValue.doubleValue() / totalCount.doubleValue());
                    Integer actualAttrValueCountForClassValue = counts.get(classValue);
                    subChiSquared += (Math.pow(actualAttrValueCountForClassValue - expectedAttrValueCountForClassValue, 2)/expectedAttrValueCountForClassValue);
                }
            }

        }

        CriticalValuesTable table = new CriticalValuesTable();
        Double chiSquaredValue = table.getChiSquaredValue(attribute.numValues(), this.confidenceLevel);

        return subChiSquared > chiSquaredValue;
    }

    private void countNumOutcomes(Instances instances, Map<Double, Integer> totalNumOfEachOutcome) {
        for (int i = 0; i < instances.size(); i++) {
            Instance instance = instances.get(i);
            Double classValue = instance.classValue();
            Integer numOutcome = totalNumOfEachOutcome.get(classValue);
            if (numOutcome == null) {
                numOutcome = 0;
            }
            totalNumOfEachOutcome.put(classValue, ++numOutcome);
        }
    }

    public void print() {
        print("", true);
    }

    private void print(String prefix, boolean isTail) {
        printThis(prefix, isTail);
        List<ID3TreeNode> children = new ArrayList<>();
        for (Double attrValue : this.children.keySet()) {
            ID3TreeNode n = this.children.get(attrValue);
            n.parentAttrValue = attrValue;
            children.add(this.children.get(attrValue));
        }
        for (int i = 0; i < children.size() - 1; i++) {
            children.get(i).print(prefix + (isTail ? "    " : "│   "), false);
        }
        if (children.size() > 0) {
            children.get(children.size() - 1)
                    .print(prefix + (isTail ?"    " : "│   "), true);
        }
    }

    public void printThis(String prefix, boolean isTail) {
        String toPrint = prefix + (isTail ? "└── " : "├── ");
        if (this.parent != null) {
            String valString = this.parent.attribute.value(this.parentAttrValue.intValue());
            toPrint += valString + " ~> ";
        }

        if (this.attribute != null) {
            toPrint += attribute.name();
        }
        else {
            toPrint += "??";
        }

        if (this.terminatedClassValue != null) {
            toPrint += " --> " + this.terminatedClassValue;
        }

        System.out.println(toPrint);
    }

    public Double evaluateInstance(Instance instance) {

        if (this.terminatedClassValue != null) {
            return this.terminatedClassValue;
        }

        // Make sure the attributes match
        String name = this.attribute.name();
        int index = 0;
        for (int i = 0; i < instance.numAttributes(); i++) {
            Attribute anAttr = instance.attribute(i);
            if (anAttr.name().equals(name)) {
                index = i;
                break;
            }
        }
        Double attributeValue = instance.value(index);
        ID3TreeNode node = this.children.get(attributeValue);
        if (node == null) {
            return 1.0;
        }
        return node.evaluateInstance(instance);
    }
}

class ID3TreeLeaf extends ID3TreeNode {

    private Double attributeValue;
    private Double classValueForAttributeValue;
    public ID3TreeLeaf(ID3TreeNode parent, Integer confidenceLevel) {
        super(parent, confidenceLevel);
    }

    public void setClassValueForAttributeValue(Double attrValue, Double classValueForAttributeValue) {
        this.attributeValue = attrValue;
        this.classValueForAttributeValue = classValueForAttributeValue;
    }

    public void printThis(String prefix, boolean isTail) {
        String valString;
        if (this.attributeValue.equals(Double.NaN)) {
            valString = "?";
        } else {
            valString = this.attribute.value(this.attributeValue.intValue());
        }
        System.out.println(prefix + (isTail ? "└── " : "├── ") + valString + " >> " + this.classValueForAttributeValue);
    }

    public Double evaluateInstance(Instance instance) {
        return this.classValueForAttributeValue;
    }
}
