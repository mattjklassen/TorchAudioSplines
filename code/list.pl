# !/usr/bin/perl

@python_files = glob("*.py");

# print "@python_files\n";

$output = "../README.txt";
$output = "> $output";

open(OUTPUT, $output) or
          die "Can't open $output.";

$firstline = "These python scripts deal with the modeling of audio segments with cubic splines.\n";
@introlines = ($firstline);
$newline = "A segment is referred to as a cycle if it occurs inside a larger segment which has\n";
push @introlines, $newline;
$newline = "approximate fundamental frequency f_0, and the cycle has approximate length 1/f_0.\n";
push @introlines, $newline;
$newline = "\n";
push @introlines, $newline;
$intro = join "", @introlines;

$firstline = "Suggested tests to run:\n\n";
@guidelines = ($firstline);
$newline = "> python torchSpline.py\n\n";
push @guidelines, $newline;
$newline = "... plots cubic spline on interval [0,1] interpolating 20 points with values in [-1,1]\n";
push @guidelines, $newline;
$newline = "equal to zero at the ends and random values in between. Edit code to change n.\n\n";
push @guidelines, $newline;
$newline = "> python wavspline.py ../audio/input.wav 200 500 20\n\n";
push @guidelines, $newline;
$newline = "... computes and plots spline for audio segment from input.wav for samples 200 to 500\n";
push @guidelines, $newline;
$newline = "with n = 20 interpolation points.\n";
push @guidelines, $newline;
$newline = "\n";
push @guidelines, $newline;
$newline = "\n";
push @guidelines, $newline;
$guide = join "", @guidelines;

$caption = "Summary of python files (in alphabetical order) with brief description of each:\n";

$description_start = "----- Brief Description -----";
$description_end = "----- ----- ----- ----- -----";

print OUTPUT "$intro";
print OUTPUT "$guide";
print OUTPUT "$caption";

$index = 1;
$writing = 0;
foreach $file (@python_files)
{
    print "$index  $file\n\n";
    print OUTPUT "\n\n$index  $file\n\n";
    open(INPUT, $file) or die "Can't open python file: $file.";
    until (eof(INPUT))
    {
       my $line = <INPUT>;
       chomp $line;
       if ($line =~ m/$description_start/) {
          $writing = 1;
       }
       if ($line =~ m/$description_end/) {
          $writing = 0;
       }
       if ($writing == 1) {
          print OUTPUT "$line\n";
       }
    }
    $index += 1;
}

close OUTPUT;

exit 0;

@headers = @hwnums;
push @headers, "AVG";
$header = join "\t", ("ID"),@headers;

$output = "quiz-scores.txt";
$output = "> $output";

$caption = "MAT 320 Quiz Scores - Fall 2022";

open(OUTPUT, $output) or
          die "Can't open $output.";

print OUTPUT "$caption\n";
print OUTPUT "$header\n";

foreach $idref (@idpairs)
{
   my @output = @$idref;
   my $sum = 0;
   my $num = 0;
   my $avg = 0;
   foreach $file (@hwfiles)
   {
      open(HW, $file) or die "Can't open hw file: $file.";
      my $added = 0;
      until (eof(HW))
      {
        my $line = <HW>;
      #  $line =~ s/,$/,0/;
        chomp $line;
        if (($line =~ m/$output[2]/) || ($line =~ m/$output[3]/)) 
        {
#	  if ($line =~ m/5245/)
#	   { print "$file: \n"; 
#	     print "the line: $line\n";  
#	   }
          my @data = split /,/, $line;
          my $score = $data[4];
          $score =~ s/[^\.0-9]//g;
#	  if ($line =~ m/5245/)
#	    { print "score: $score\n"; }
          $test_numeric = $score;
          $test_numeric =~ s/[^0-9]//g;
          if ($test_numeric eq "") {
             next; 
          }
          $added = 1;
          push @output, $score;    
          $sum = $sum + $score;
          $num++;
          $avg = $sum / $num;
        }
      }
      if ($added == 0)
      {
        push @output, "0";
          $num++;
          $avg = $sum / $num;
      }
      close HW;    
      }
   # pull off first name, last name and email id
   shift @output;
   shift @output;
   shift @output;
   my $newavg = sprintf "%2.2f", $avg;
   push @output, $newavg;
   $lineout = join "\t", @output;
   push @outlines, $lineout;
}

@outsort = sort @outlines;

foreach $outline (@outsort)
{
   print OUTPUT "$outline\n";
}


close OUTPUT;





exit 0;


