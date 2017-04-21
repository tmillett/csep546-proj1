package com.company;

import weka.core.Attribute;

/**
 * Created by tmillett on 4/14/17.
 */
public class AttrInfo {

    private Attribute attribute;
    private Double value;
    private Integer count;
    public AttrInfo() {

    }

    public Attribute getAttribute() {
        return attribute;
    }

    public void setAttribute(Attribute attribute) {
        this.attribute = attribute;
    }

    public Double getValue() {
        return value;
    }

    public void setValue(Double value) {
        this.value = value;
    }

    public Integer getCount() {
        return count;
    }

    public void setCount(Integer count) {
        this.count = count;
    }

    public String toString() {

        String valString = this.attribute.value(this.value.intValue());

        return this.attribute.name() + " " + valString + " " + this.count;
    }

}
