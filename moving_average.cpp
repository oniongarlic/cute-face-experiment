#include "moving_average.hpp"

MovingAverage::MovingAverage(int n)
{
    m_n=n;
}

double MovingAverage::add(double val)
{
    if (m_v==0)
        m_value=val;
    m_v++;
    double d=(val-m_value)/(double)m_v;
    m_value=m_value+d;

    return m_value;
}

double MovingAverage::get()
{
    return m_value;
}
