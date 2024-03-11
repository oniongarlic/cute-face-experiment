
class MovingAverage // sort-of
{
public:
	MovingAverage(int n=5);
	double add(double val);
	double get();

private:
	double m_value=0;
	int m_v=0;
	int m_n;
};
