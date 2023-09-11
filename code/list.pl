# !/usr/bin/perl

sub listcmp {
    my ($aref, $bref) = @_;
    my $i = 0;
    my $s = 0;
	foreach $a (@$aref) {
   	   if ($a =~ m/$$bref[$i]/) {
		$s = $s+1;	
	   }		
	   $i++;
	}
    return $s;
}

# $idfile = "ids.csv";
# 
# open(IDS, $idfile) or 
#           die "Can't open $input.";
# 
# until (eof(IDS))
#    {
# 	my $line = <IDS>;
#         chomp $line;
#         push @idlines, $line;
#         @id = split /,/, $line;
#         push @idpairs, [ @id ];
# #        print "@id\n";
#    }
# 
# close IDS;

@python_files = glob("*.py");

print "@python_files\n";
$index = 0;

# foreach $file (@python_files)
# {
#     print "$index  $file\n";
#     $index += 1;
# }

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


